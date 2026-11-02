import csv
import random
from datetime import datetime, timedelta

def generate_sample_sales():
    categories = {
        "Enterprise Server Solutions": {"price": (1200, 4500), "weight": 0.35},
        "Cloud Storage Appliances": {"price": (600, 1800), "weight": 0.25},
        "Cybersecurity Peripherals": {"price": (250, 950), "weight": 0.20},
        "Networking Cables & Adapters": {"price": (15, 65), "weight": 0.12},
        "Legacy Replacement Parts": {"price": (20, 80), "weight": 0.08} # Underperforming category
    }
    
    regions = ["North America", "EMEA", "APAC", "LATAM"]
    channels = ["Direct Enterprise", "Partner Reseller", "Digital Self-Serve"]
    
    start_date = datetime(2025, 1, 1)
    rows = []
    
    for i in range(1, 1201):
        date_val = start_date + timedelta(days=random.randint(0, 365))
        # Choose category based on distribution
        cat_choices = list(categories.keys())
        cat_weights = [categories[c]["weight"] for c in cat_choices]
        chosen_cat = random.choices(cat_choices, weights=cat_weights, k=1)[0]
        
        unit_price = round(random.uniform(*categories[chosen_cat]["price"]), 2)
        units_sold = random.randint(1, 15) if unit_price > 500 else random.randint(10, 80)
        
        # In underperforming category, keep unit sales very low
        if chosen_cat == "Legacy Replacement Parts":
            units_sold = random.randint(1, 4)
            
        revenue = round(unit_price * units_sold, 2)
        margin_pct = random.uniform(0.18, 0.45)
        profit = round(revenue * margin_pct, 2)
        
        rows.append({
            "order_id": f"ORD-{10000 + i}",
            "order_date": date_val.strftime("%Y-%m-%d"),
            "category": chosen_cat,
            "region": random.choice(regions),
            "sales_channel": random.choice(channels),
            "units_sold": units_sold,
            "unit_price": unit_price,
            "revenue": revenue,
            "profit": profit
        })

    # Sort by date
    rows.sort(key=lambda x: x["order_date"])

    with open("c:/Users/DELL/Desktop/aura/data/sample_sales.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Generated {len(rows)} sales records into data/sample_sales.csv")

if __name__ == "__main__":
    generate_sample_sales()
