"""
mem_drive.py

Contains logic for large np.memmap usage, Drive uploads/backups, etc.
"""

import os
import io
import datetime
import numpy as np
import logging
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from pathlib import Path

logger = logging.getLogger("MemoryDriveManager")
logger.setLevel(logging.INFO)

class MemoryDriveManager:
    def __init__(self, memory_size=50_000_000_000, local_memory_size=10_000_000_000):
        self.memory_size = memory_size
        self.local_memory_size = local_memory_size
        self.memory = None
        self.drive_service = None
        self.last_backup_date = None
        self.memory_file_path = "brain_memory.bin"
        self.memory_shape = (int(memory_size / 8), )  # 8 bytes per float64
        # Google Drive related settings
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.token_path = 'token.json'
        self.credentials_path = 'credentials.json'
        self.backup_folder_id = None

    def initialize_memory(self):
        """Create or load a memory-mapped numpy array for efficient large-scale storage."""
        try:
            # Check if memory file exists
            if os.path.exists(self.memory_file_path):
                logger.info(f"Loading existing memory file: {self.memory_file_path}")
                # Load existing memory file
                self.memory = np.memmap(
                    self.memory_file_path,
                    dtype=np.float64,
                    mode='r+',
                    shape=self.memory_shape
                )
            else:
                logger.info(f"Creating new memory file: {self.memory_file_path}")
                # Create new memory file
                self.memory = np.memmap(
                    self.memory_file_path,
                    dtype=np.float64,
                    mode='w+',
                    shape=self.memory_shape
                )
                # Initialize with zeros
                self.memory[:] = 0
                self.memory.flush()
            
            logger.info(f"Memory initialized with shape: {self.memory.shape}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize memory: {str(e)}")
            # Create a smaller fallback memory if full size fails
            try:
                fallback_size = min(self.local_memory_size, 1_000_000_000)  # 1GB or smaller
                fallback_shape = (int(fallback_size / 8), )
                logger.info(f"Attempting fallback memory with size: {fallback_size}")
                self.memory = np.memmap(
                    "fallback_" + self.memory_file_path,
                    dtype=np.float64,
                    mode='w+',
                    shape=fallback_shape
                )
                self.memory[:] = 0
                self.memory.flush()
                return True
            except Exception as fallback_error:
                logger.error(f"Fallback memory initialization failed: {str(fallback_error)}")
                return False

    def initialize_drive_service(self):
        """Initialize Google Drive service with proper authentication."""
        try:
            creds = None
            # Check if token file exists
            if os.path.exists(self.token_path):
                creds = Credentials.from_authorized_user_info(
                    json.loads(open(self.token_path, 'r').read()),
                    self.scopes
                )
            
            # If credentials don't exist or are invalid, refresh or get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    # If credentials.json doesn't exist, warn and return
                    if not os.path.exists(self.credentials_path):
                        logger.warning(f"Credentials file not found: {self.credentials_path}")
                        return False
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, self.scopes)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for next run
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())
            
            # Build the Drive service
            self.drive_service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive service initialized successfully")
            
            # Create or find backup folder
            self._setup_backup_folder()
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize Drive service: {str(e)}")
            return False

    def _setup_backup_folder(self):
        """Create or find the backup folder in Google Drive."""
        if not self.drive_service:
            logger.warning("Drive service not initialized")
            return
        
        folder_name = "KairoMind_Backups"
        
        # Check if folder already exists
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.drive_service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        
        if items:
            self.backup_folder_id = items[0]['id']
            logger.info(f"Found existing backup folder: {self.backup_folder_id}")
        else:
            # Create new folder
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.drive_service.files().create(body=file_metadata, fields='id').execute()
            self.backup_folder_id = folder.get('id')
            logger.info(f"Created new backup folder: {self.backup_folder_id}")

    def upload_to_drive(self, file_path, file_name=None):
        """Upload a file to Google Drive."""
        if not self.drive_service:
            logger.warning("Drive service not initialized")
            return None
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            # Use original filename if not specified
            if file_name is None:
                file_name = os.path.basename(file_path)
            
            # Prepare metadata
            file_metadata = {
                'name': file_name,
                'parents': [self.backup_folder_id] if self.backup_folder_id else []
            }
            
            # Create media
            media = MediaFileUpload(
                file_path,
                resumable=True
            )
            
            # Create the file
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            logger.info(f"File uploaded: {file_name}, ID: {file.get('id')}")
            return file.get('id')
        
        except Exception as e:
            logger.error(f"Failed to upload file {file_path}: {str(e)}")
            return None

    def download_from_drive(self, file_id, file_name):
        """Download a file from Google Drive."""
        if not self.drive_service:
            logger.warning("Drive service not initialized")
            return False
        
        try:
            # Get file metadata
            file_metadata = self.drive_service.files().get(fileId=file_id).execute()
            
            # Create download request
            request = self.drive_service.files().get_media(fileId=file_id)
            
            # Create a BytesIO stream to save the file content
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            # Download the file
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.info(f"Download progress: {int(status.progress() * 100)}%")
            
            # Save the file
            with open(file_name, 'wb') as f:
                f.write(file_content.getvalue())
            
            logger.info(f"File downloaded: {file_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {str(e)}")
            return False

    def create_backup(self, frequency_days=7):
        """Create a backup of the memory file if enough time has passed."""
        if not self.drive_service:
            logger.warning("Drive service not initialized")
            return False
        
        # Check if a backup is needed
        current_date = datetime.datetime.now()
        if self.last_backup_date:
            days_since_backup = (current_date - self.last_backup_date).days
            if days_since_backup < frequency_days:
                logger.info(f"Backup not needed yet. Days since last backup: {days_since_backup}")
                return False
        
        # Ensure memory is flushed to disk
        if self.memory is not None:
            self.memory.flush()
        
        # Create timestamped backup name
        timestamp = current_date.strftime("%Y%m%d_%H%M%S")
        backup_name = f"brain_backup_{timestamp}.bin"
        
        # Upload memory file
        file_id = self.upload_to_drive(self.memory_file_path, backup_name)
        
        if file_id:
            self.last_backup_date = current_date
            logger.info(f"Backup created: {backup_name}")
            
            # Clean up old backups
            self._manage_backups(max_backups=5)
            return True
        else:
            logger.error("Backup creation failed")
            return False

    def adaptive_upload(self, data_chunk, chunk_index, total_chunks):
        """Upload a chunk of data with adaptive retry logic."""
        if not self.drive_service:
            logger.warning("Drive service not initialized")
            return False
        
        # Create a temporary file for the chunk
        chunk_file = f"temp_chunk_{chunk_index}.bin"
        
        try:
            # Save the chunk to a temporary file
            with open(chunk_file, 'wb') as f:
                np.save(f, data_chunk)
            
            # Set up exponential backoff for retries
            max_retries = 5
            retry_wait = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    chunk_name = f"chunk_{chunk_index}_of_{total_chunks}.bin"
                    file_id = self.upload_to_drive(chunk_file, chunk_name)
                    
                    if file_id:
                        logger.info(f"Chunk {chunk_index}/{total_chunks} uploaded successfully")
                        return file_id
                    else:
                        raise Exception("Upload returned no file ID")
                
                except Exception as retry_error:
                    logger.warning(f"Upload attempt {attempt+1} failed: {str(retry_error)}")
                    if attempt < max_retries - 1:
                        wait_time = retry_wait * (2 ** attempt)
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to upload chunk {chunk_index}: {str(e)}")
            return False
        
        finally:
            # Clean up temporary file
            if os.path.exists(chunk_file):
                os.remove(chunk_file)

    def _manage_backups(self, max_backups=5):
        """Maintain a limited number of backups, removing oldest files if needed."""
        if not self.drive_service or not self.backup_folder_id:
            return
        
        try:
            # Get all backup files
            query = f"'{self.backup_folder_id}' in parents and trashed=false"
            results = self.drive_service.files().list(
                q=query,
                orderBy='createdTime',
                fields="files(id, name, createdTime)"
            ).execute()
            
            backups = results.get('files', [])
            
            # Delete oldest backups if we have too many
            if len(backups) > max_backups:
                # Sort by creation time, oldest first
                backups.sort(key=lambda x: x['createdTime'])
                
                # Delete oldest files
                files_to_delete = backups[:-max_backups]
                for file in files_to_delete:
                    self.drive_service.files().delete(fileId=file['id']).execute()
                    logger.info(f"Deleted old backup: {file['name']}")
        
        except Exception as e:
            logger.error(f"Error managing backups: {str(e)}")

    def store_embeddings(self, embeddings, indices=None):
        """Store embedding vectors in the memory map."""
        if self.memory is None:
            logger.error("Memory not initialized")
            return False
        
        try:
            if indices is None:
                # Find empty space (zeros) in memory
                zero_indices = np.where(self.memory[:len(embeddings)] == 0)[0]
                if len(zero_indices) >= len(embeddings):
                    indices = zero_indices[:len(embeddings)]
                else:
                    # Not enough space at the beginning, try to find elsewhere
                    all_zeros = np.where(self.memory == 0)[0]
                    if len(all_zeros) >= len(embeddings):
                        indices = all_zeros[:len(embeddings)]
                    else:
                        # No empty space, overwrite oldest data (simple approach)
                        indices = np.arange(len(embeddings))
            
            # Store embeddings at the specified indices
            self.memory[indices] = embeddings
            self.memory.flush()
            
            logger.info(f"Stored {len(embeddings)} embeddings at indices {indices[0]}-{indices[-1]}")
            return indices
        
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            return False

    def retrieve_embeddings(self, indices):
        """Retrieve embedding vectors from the memory map."""
        if self.memory is None:
            logger.error("Memory not initialized")
            return None
        
        try:
            embeddings = self.memory[indices]
            logger.info(f"Retrieved {len(indices)} embeddings")
            return embeddings
        
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings: {str(e)}")
            return None

    def search_similar_embeddings(self, query_embedding, top_k=5):
        """Find similar embeddings using cosine similarity."""
        if self.memory is None:
            logger.error("Memory not initialized")
            return None
        
        try:
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            
            # Calculate dot product with all non-zero embeddings
            non_zero_indices = np.where(self.memory != 0)[0]
            if len(non_zero_indices) == 0:
                return []
            
            # Get non-zero embeddings
            embeddings = self.memory[non_zero_indices]
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms>0)
            
            # Calculate similarities
            similarities = np.dot(normalized, query_embedding)
            
            # Get top-k indices
            if len(similarities) < top_k:
                top_k = len(similarities)
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Map back to original indices
            result_indices = non_zero_indices[top_indices]
            
            return {
                'indices': result_indices,
                'similarities': similarities[top_indices]
            }
        
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {str(e)}")
            return None

import json  # Add this at the top of the file
import time  # Add this at the top of the file
