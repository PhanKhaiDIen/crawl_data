import requests
import csv
import time
import random
import os
from urllib.parse import urljoin

# Configuration
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://tiki.vn/chuot-choi-game/c3428',
}
IMAGE_DIR = 'product_images'
MAX_PAGES = 3
PRODUCTS_PER_PAGE = 17
DELAY_RANGE = (1.0, 2.5)

def setup_image_directory():
    os.makedirs(IMAGE_DIR, exist_ok=True)

def download_and_convert_image(image_url, product_id):
    if not image_url or not product_id:
        return ""
    
    try:
        # Standardize URL
        if image_url.startswith('//'):
            image_url = 'https:' + image_url
        
        # Set filename
        filename = f"{product_id}.jpg"
        filepath = os.path.join(IMAGE_DIR, filename)
        
        # Download image
        response = requests.get(image_url, headers=HEADERS, stream=True, timeout=10)
        response.raise_for_status()
        
        # Save as JPG
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        
        return filename
        
    except Exception as e:
        print(f"⚠️ Failed to download image for product {product_id}: {str(e)}")
        return ""

def crawl_mice_from_category():
    setup_image_directory()
    all_products = []
    category_name = "Chuột chơi game"
    category_id = 3428

    for page in range(1, MAX_PAGES + 1):
        print(f"📊 Collecting page {page}/{MAX_PAGES}...")

        try:
            response = requests.get(
                "https://tiki.vn/api/v2/products",
                headers=HEADERS,
                params={
                    "limit": PRODUCTS_PER_PAGE,
                    "category": category_id,
                    "page": page,
                    "sort": "newest"
                },
                timeout=15
            )

            if response.status_code == 200:
                products = response.json().get("data", [])
                
                if not products:
                    print(f"⏩ No products on page {page}, stopping collection")
                    break

                for product in products:
                    product_id = str(product.get('id', ''))
                    thumbnail_url = product.get('thumbnail_url', '')
                    
                   
                    image_filename = download_and_convert_image(thumbnail_url, product_id)
                    
                    all_products.append({
                        "product_id": product_id,
                        "category": category_name,
                        "product_name": product.get('name', ''),
                        "price": product.get('price', 0),
                        "image": image_filename,  # Saved filename instead of URL
                        "image_url": thumbnail_url,  # Keep original URL for reference
                        "source": f"https://tiki.vn/{product.get('url_path', '')}"
                    })

                print(f"✅ Added {len(products)} products from page {page}")
            else:
                print(f"⚠️ Error {response.status_code} on page {page}")
                break

        except Exception as e:
            print(f"❌ Connection error: {str(e)}")
            break

        time.sleep(random.uniform(*DELAY_RANGE))

    return all_products

def save_to_csv(products, filename="tiki_gaming_mice.csv"):
    """Save product data to CSV file"""
    if not products:
        print("⛔ No products to save")
        return

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["product_id", "category", "product_name", "price", "image", "image_url", "source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(products)
    
    print(f"💾 Saved {len(products)} products to {filename}")

if __name__ == "__main__":
    print("🛒 Starting to collect gaming mice from Tiki...")
    products = crawl_mice_from_category()
    
    if products:
        save_to_csv(products)
        
        # Print sample product
        sample = products[0]
        print("\nSample product:")
        print(f"ID: {sample['product_id']}")
        print(f"Name: {sample['product_name']}")
        print(f"Price: {sample['price']:,} VND")
        print(f"Image file: {sample['image']}")
        print(f"Image URL: {sample['image_url'][:50]}...")
        print(f"Link: {sample['source']}")