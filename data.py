import pandas as pd
import random
from datetime import date, timedelta

random.seed(42)

CLIENTS = [
    "NordicFit",
    "OsloEats",
    "BergenStyle",
    "TromsøTech",
    "StavangerHome",
    "TrondheimTravel",
    "KristiansandKids",
]

CAMPAIGNS = {
    "NordicFit": ["Summer Body Challenge", "Protein Launch"],
    "OsloEats": ["Weekend Brunch", "Vegan Menu Promo", "Lunch Deal"],
    "BergenStyle": ["Fall Collection", "Flash Sale"],
    "TromsøTech": ["Back to School", "Enterprise SaaS Push", "Holiday Tech"],
    "StavangerHome": ["Interior Refresh", "Black Friday"],
    "TrondheimTravel": ["Northern Lights Tour", "Budget Escape"],
    "KristiansandKids": ["Back to School Kids", "Summer Camp"],
}

CHANNELS = ["Google Ads", "Meta Ads", "TikTok Ads"]

GOALS = ["Brand Awareness", "Lead Generation", "Direct Sales", "App Installs"]

AUDIENCES = [
    "18-34 fitness enthusiasts",
    "25-45 urban professionals",
    "30-55 homeowners",
    "18-24 students",
    "35-55 families",
    "20-35 tech early adopters",
    "25-40 foodies",
    "18-30 fashion-forward women",
    "40-60 travel enthusiasts",
    "parents with children under 12",
]

AD_TEXTS = [
    "Transform your body in 30 days. Join thousands of Norwegians getting fit!",
    "Fuel your performance with our premium protein blends. Free shipping today.",
    "Oslo's best brunch — now bookable online. Reserve your table this weekend.",
    "100% plant-based. 100% delicious. Explore our new vegan menu.",
    "Fresh lunch deals from NOK 89. Order now and skip the queue.",
    "New fall arrivals just dropped. Discover Scandinavian fashion at its finest.",
    "Flash sale — 40% off everything. 24 hours only. Don't miss out.",
    "The laptop students trust. Back to school deals starting from NOK 4999.",
    "Automate your workflow. Try our enterprise platform free for 30 days.",
    "The perfect holiday gift for the tech lover in your life.",
    "Refresh your home this season. Handpicked Scandinavian interiors.",
    "Black Friday starts now. Up to 60% off furniture and decor.",
    "Chase the Northern Lights. Guided tours from Tromsø — book now.",
    "Escape the city from NOK 1499. Weekend getaways across Norway.",
    "Get your kids ready for school. Backpacks, stationery, and more.",
    "Summer camp 2026 — adventure, learning & fun. Limited spots available.",
]


def _channel_multipliers(channel: str) -> dict:
    """Rough CPM/CTR/CVR profile per channel."""
    if channel == "Google Ads":
        return {"cpm": 120, "ctr": 0.045, "cvr": 0.06, "roas_base": 3.5}
    elif channel == "Meta Ads":
        return {"cpm": 80, "ctr": 0.025, "cvr": 0.04, "roas_base": 2.8}
    else:  # TikTok
        return {"cpm": 55, "ctr": 0.018, "cvr": 0.025, "roas_base": 2.0}


def generate_dataset() -> pd.DataFrame:
    rows = []
    start_date = date(2025, 10, 6)  # Week 1 starts Oct 6 2025

    for client in CLIENTS:
        campaigns = CAMPAIGNS[client]
        # Pick 2–3 channels per campaign
        for campaign in campaigns:
            n_channels = random.randint(2, 3)
            channels = random.sample(CHANNELS, n_channels)
            n_weeks = random.randint(4, 8)
            goal = random.choice(GOALS)
            audience = random.choice(AUDIENCES)
            ad_text = random.choice(AD_TEXTS)

            for channel in channels:
                m = _channel_multipliers(channel)
                for week_num in range(1, n_weeks + 1):
                    week_date = start_date + timedelta(weeks=week_num - 1)

                    # Spend with some noise and a slight ramp-up
                    base_spend = random.uniform(800, 5000)
                    ramp = 1 + 0.05 * (week_num - 1)
                    noise = random.uniform(0.8, 1.2)
                    spend = round(base_spend * ramp * noise, 2)

                    impressions = int(spend / m["cpm"] * 1000 * random.uniform(0.9, 1.1))
                    clicks = int(impressions * m["ctr"] * random.uniform(0.85, 1.15))
                    conversions = int(clicks * m["cvr"] * random.uniform(0.7, 1.3))
                    revenue = round(conversions * random.uniform(200, 1200) * random.uniform(0.9, 1.1), 2)

                    roas = round(revenue / spend, 2) if spend > 0 else 0.0
                    ctr = round(clicks / impressions * 100, 2) if impressions > 0 else 0.0

                    rows.append({
                        "client": client,
                        "campaign": campaign,
                        "channel": channel,
                        "week": week_num,
                        "week_date": week_date.strftime("%Y-%m-%d"),
                        "spend": spend,
                        "impressions": impressions,
                        "clicks": clicks,
                        "conversions": conversions,
                        "revenue": revenue,
                        "roas": roas,
                        "ctr": ctr,
                        "ad_text": ad_text,
                        "audience": audience,
                        "goal": goal,
                    })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    print("\nClients:", df["client"].unique())
    print("Channels:", df["channel"].unique())
