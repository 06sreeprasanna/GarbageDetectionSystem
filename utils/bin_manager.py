import os

def get_bin_name(category):
    """
    Map waste categories to bins with enhanced logic
    """
    if not category:
        return None
    
    category = str(category).lower().strip()
    
    # Biodegradable waste (Green bin)
    biodegradable = ['organic', 'food', 'vegetable', 'fruit', 'compost']
    if any(bio in category for bio in biodegradable):
        return 'Green'
    
    # Recyclable waste (Blue bin)
    recyclable = ['paper', 'cardboard', 'metal', 'aluminum', 'glass', 
                 'plastic', 'pet', 'hdpe', 'bottle', 'can']
    if any(rec in category for rec in recyclable):
        return 'Blue'
    
    # Hazardous waste (Red bin)
    hazardous = ['battery', 'chemical', 'medical', 'electronic', 'e-waste']
    if any(haz in category for haz in hazardous):
        return 'Red'
    
    # Non-recyclable waste (default to Red)
    return 'Red'

def get_bin_status(bin_folders, max_images=20):
    status = {}
    for bin_name, path in bin_folders.items():
        try:
            files = [f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
            is_full = len(files) >= max_images
            status[bin_name] = {
                'count': len(files),
                'full': is_full
            }
        except Exception as e:
            status[bin_name] = {
                'count': 0,
                'full': False,
                'error': str(e)
            }
    return status
