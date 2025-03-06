"""
contacts_integration.py

Display contacts in HUD panel using Google Contacts API (via People API)
"""

import logging
import datetime

logger = logging.getLogger("ContactsIntegration")
logger.setLevel(logging.INFO)

class ContactsIntegration:
    def __init__(self, api_manager):
        """
        Initialize Google Contacts integration
        
        Args:
            api_manager: GoogleAPIManager instance
        """
        self.api_manager = api_manager
        self.contacts_service = None
        self.cached_contacts = None
        self.cache_time = None
    
    def initialize(self):
        """Initialize the Contacts service (People API)"""
        self.contacts_service = self.api_manager.get_service('people')
        return self.contacts_service is not None
    
    def get_contacts(self, max_results=100, force_refresh=False):
        """Get list of contacts"""
        try:
            if not self.contacts_service:
                if not self.initialize():
                    return []
            
            # Return cached contacts if available and not expired
            if (not force_refresh and self.cached_contacts and self.cache_time and 
                (datetime.datetime.now() - self.cache_time).total_seconds() < 3600):
                return self.cached_contacts
            
            # Request contacts from API
            contacts = []
            page_token = None
            
            while True and len(contacts) < max_results:
                # Make the API call
                results = self.contacts_service.people().connections().list(
                    resourceName='people/me',
                    pageSize=min(1000, max_results - len(contacts)),
                    pageToken=page_token,
                    personFields='names,emailAddresses,phoneNumbers,photos,organizations,addresses',
                    sortOrder='LAST_MODIFIED_DESCENDING'
                ).execute()
                
                # Process contacts
                if 'connections' in results:
                    for person in results['connections']:
                        contact = self._process_contact(person)
                        if contact:  # Only add if we have a valid contact
                            contacts.append(contact)
                
                # Check for more pages
                if 'nextPageToken' in results and len(contacts) < max_results:
                    page_token = results['nextPageToken']
                else:
                    break
            
            # Cache results
            self.cached_contacts = contacts
            self.cache_time = datetime.datetime.now()
            
            return contacts
            
        except Exception as e:
            logger.error(f"Error getting contacts: {e}")
            return []
    
    def search_contacts(self, query):
        """Search contacts by name, email, or phone"""
        try:
            if not self.contacts_service:
                if not self.initialize():
                    return []
            
            # Get all contacts first (use cache if available)
            all_contacts = self.get_contacts(max_results=2000)
            
            # Filter contacts locally
            query = query.lower()
            filtered_contacts = []
            
            for contact in all_contacts:
                # Check name
                if 'display_name' in contact and query in contact['display_name'].lower():
                    filtered_contacts.append(contact)
                    continue
                
                # Check emails
                if 'emails' in contact:
                    if any(query in email.lower() for email in contact['emails']):
                        filtered_contacts.append(contact)
                        continue
                
                # Check phones
                if 'phones' in contact:
                    if any(query in phone.lower() for phone in contact['phones']):
                        filtered_contacts.append(contact)
                        continue
                
                # Check organizations
                if 'organization' in contact:
                    if query in contact['organization'].lower():
                        filtered_contacts.append(contact)
                        continue
            
            return filtered_contacts
            
        except Exception as e:
            logger.error(f"Error searching contacts: {e}")
            return []
    
    def get_contact_details(self, resource_name):
        """Get detailed information for a specific contact"""
        try:
            if not self.contacts_service:
                if not self.initialize():
                    return None
            
            # Make the API call
            person = self.contacts_service.people().get(
                resourceName=resource_name,
                personFields='names,emailAddresses,phoneNumbers,photos,organizations,addresses,birthdays,urls,biographies,userDefined'
            ).execute()
            
            return self._process_contact(person, detailed=True)
            
        except Exception as e:
            logger.error(f"Error getting contact details: {e}")
            return None
    
    def _process_contact(self, person, detailed=False):
        """Process a contact from the API into a simplified format"""
        try:
            resource_name = person.get('resourceName', '')
            
            # Skip if no resource name
            if not resource_name:
                return None
            
            # Get name
            display_name = 'Unnamed Contact'
            if 'names' in person and person['names']:
                display_name = person['names'][0].get('displayName', 'Unnamed Contact')
            
            # Initialize contact object
            contact = {
                'resource_name': resource_name,
                'id': resource_name.split('/')[-1],
                'display_name': display_name
            }
            
            # Get emails
            if 'emailAddresses' in person and person['emailAddresses']:
                contact['emails'] = []
                primary_email = None
                
                for email in person['emailAddresses']:
                    email_value = email.get('value')
                    if email_value:
                        contact['emails'].append(email_value)
                        
                        # Check if primary
                        if 'metadata' in email and email['metadata'].get('primary'):
                            primary_email = email_value
                
                # Add primary email
                if primary_email:
                    contact['primary_email'] = primary_email
                else:
                    contact['primary_email'] = contact['emails'][0]
            
            # Get phones
            if 'phoneNumbers' in person and person['phoneNumbers']:
                contact['phones'] = []
                primary_phone = None
                
                for phone in person['phoneNumbers']:
                    phone_value = phone.get('value')
                    if phone_value:
                        contact['phones'].append(phone_value)
                        
                        # Check if primary
                        if 'metadata' in phone and phone['metadata'].get('primary'):
                            primary_phone = phone_value
                
                # Add primary phone
                if primary_phone:
                    contact['primary_phone'] = primary_phone
                else:
                    contact['primary_phone'] = contact['phones'][0]
            
            # Get photo
            if 'photos' in person and person['photos']:
                for photo in person['photos']:
                    if 'url' in photo:
                        contact['photo_url'] = photo['url']
                        break
            
            # Get organization
            if 'organizations' in person and person['organizations']:
                org = person['organizations'][0]
                org_parts = []
                
                if 'name' in org:
                    org_parts.append(org['name'])
                
                if 'title' in org:
                    org_parts.append(org['title'])
                
                if org_parts:
                    contact['organization'] = ' - '.join(org_parts)
            
            # For detailed view, add more information
            if detailed:
                # Add addresses
                if 'addresses' in person and person['addresses']:
                    contact['addresses'] = []
                    
                    for address in person['addresses']:
                        formatted_address = address.get('formattedValue', '')
                        if formatted_address:
                            address_type = address.get('type', '')
                            contact['addresses'].append({
                                'type': address_type,
                                'value': formatted_address
                            })
                
                # Add birthdays
                if 'birthdays' in person and person['birthdays']:
                    for birthday in person['birthdays']:
                        if 'date' in birthday:
                            date = birthday['date']
                            if 'year' in date and 'month' in date and 'day' in date:
                                contact['birthday'] = f"{date['year']}-{date['month']:02d}-{date['day']:02d}"
                            elif 'month' in date and 'day' in date:
                                contact['birthday'] = f"{date['month']:02d}-{date['day']:02d}"
                            break
                
                # Add websites
                if 'urls' in person and person['urls']:
                    contact['websites'] = []
                    
                    for url in person['urls']:
                        url_value = url.get('value', '')
                        if url_value:
                            url_type = url.get('type', '')
                            contact['websites'].append({
                                'type': url_type,
                                'value': url_value
                            })
                
                # Add biography
                if 'biographies' in person and person['biographies']:
                    for bio in person['biographies']:
                        bio_value = bio.get('value', '')
                        if bio_value:
                            contact['biography'] = bio_value
                            break
                
                # Add custom fields
                if 'userDefined' in person and person['userDefined']:
                    contact['custom_fields'] = []
                    
                    for field in person['userDefined']:
                        key = field.get('key', '')
                        value = field.get('value', '')
                        if key and value:
                            contact['custom_fields'].append({
                                'key': key,
                                'value': value
                            })
            
            return contact
            
        except Exception as e:
            logger.error(f"Error processing contact: {e}")
            return None
    
    def get_frequent_contacts(self, count=10):
        """Get most frequently contacted people for quick access"""
        try:
            all_contacts = self.get_contacts(max_results=500)
            
            # For this demo, we'll just return some random contacts
            # In a real implementation, you'd need to track contact frequency
            import random
            if len(all_contacts) > count:
                return random.sample(all_contacts, count)
            else:
                return all_contacts
                
        except Exception as e:
            logger.error(f"Error getting frequent contacts: {e}")
            return []

# Example usage
if __name__ == "__main__":
    from google_api_auth import GoogleAPIManager
    
    # Initialize API manager
    api_manager = GoogleAPIManager()
    if api_manager.authenticate():
        # Initialize Contacts integration
        contacts = ContactsIntegration(api_manager)
        
        # Get contacts
        all_contacts = contacts.get_contacts(max_results=10)
        print(f"Found {len(all_contacts)} contacts")
        
        # Print first few contacts
        for contact in all_contacts[:5]:
            print(f"- {contact['display_name']}")
            if 'primary_email' in contact:
                print(f"  Email: {contact['primary_email']}")
            if 'primary_phone' in contact:
                print(f"  Phone: {contact['primary_phone']}")
            print()
        
        # Search for a contact
        if all_contacts:
            search_term = all_contacts[0]['display_name'].split()[0]
            search_results = contacts.search_contacts(search_term)
            print(f"Search for '{search_term}' found {len(search_results)} contacts")
