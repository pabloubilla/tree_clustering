import pandas as pd
import requests

# Example DataFrame
data = {'Species Name': ['Lion', 'Tiger', 'Bear']}
df = pd.DataFrame(data)

# Bing API details
api_key = 'YOUR_BING_API_KEY_HERE'
endpoint = 'https://api.bing.microsoft.com/v7.0/images/search'

# Function to fetch the image URL
def get_image_url(species):
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    params = {'q': species + ' species', 'count': 1}  # Fetch only the first image

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()  # Raises a HTTPError for bad responses
        search_results = response.json()
        first_image_url = search_results['value'][0]['contentUrl']
        return first_image_url
    except Exception as e:
        print(f"Error fetching image for {species}: {e}")
        return None

# Apply the function to the dataframe
df['IMG_URL'] = df['Species Name'].apply(get_image_url)
print(df)
