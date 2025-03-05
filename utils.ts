/**
 * Utility functions for Hextrix AI
 */

/**
 * Convert a File object to base64
 * @param {File} file - The file to convert
 * @returns {string} - Base64 encoded file
 */
export async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
  });
}