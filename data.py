import pandas as pd
import random
from datetime import date, timedelta

random.seed(7391)  # project-specific seed — reproducible but unique

CLIENTS = [
    "Ra Solar",
    "Osiris Organics",
    "Isis Apparel",
    "Horus Media",
    "Anubis Analytics",
    "Thoth Technology",
    "Bastet Beauty",
    "Hathor Hospitality",
    "Khnum Craft",
    "Sekhmet Sport",
]

CAMPAIGNS = {
    "Ra Solar":           ["New Year Reset", "Spring Detox", "Marathon Prep"],
    "Osiris Organics":    ["Winter Capsule", "Outlet Weekend", "Influencer Drop"],
    "Isis Apparel":       ["Kitchen Remodel", "Smart Home Launch", "Spring Clean"],
    "Horus Media":        ["Streetwear Release", "Back to Basics", "Collab Drop"],
    "Anubis Analytics":   ["Cabin Season", "Holiday Decor", "Summer Refresh"],
    "Thoth Technology":   ["Healthy Snack Box", "Office Bundle", "Student Deal"],
    "Bastet Beauty":      ["Podcast Ad Push", "Newsletter Growth", "Creator Fund"],
    "Hathor Hospitality": ["Outdoor Season", "Winter Sports", "Gear Clearance"],
    "Khnum Craft":        ["Loyalty Launch", "Weekend Flash", "Member Perks"],
    "Sekhmet Sport":      ["Trail Running Club", "Gym Opening", "Team Kits"],
}

CHANNELS = ["Google Ads", "Meta Ads", "TikTok Ads"]

GOALS = ["Brand Awareness", "Lead Generation", "Direct Sales", "App Installs"]

AUDIENCES = [
    "18-30 urban wellness seekers",
    "25-40 mythology enthusiasts",
    "22-35 streetwear and culture fans",
    "30-50 sustainability-conscious shoppers",
    "18-28 health-focused students",
    "35-55 premium lifestyle seekers",
    "20-32 social media creatives",
    "28-45 remote professionals",
    "40-60 home improvement planners",
    "18-24 Gen Z trend adopters",
    "30-45 working parents",
    "50-65 culturally engaged retirees",
]

AD_TEXTS = [
    "Harness the power of the sun. Ra Solar brings clean energy to every home.",
    "Dawn is yours. Switch to solar and cut your bills from day one.",
    "Illuminate every room — and your future. Solar panels, installed in a day.",
    "Ancient wisdom, modern nutrition. Osiris Organics — grown with intention.",
    "Detox your routine. Monthly wellness boxes inspired by the Nile valley.",
    "Eat like royalty. Organic bundles delivered to your door every week.",
    "Wear your mythology. Isis Apparel — limited pieces, infinite story.",
    "The goddess drop has arrived. Sacred-inspired fashion, available now.",
    "Two icons. One collection. The collab the culture has been waiting for.",
    "Be seen from every angle. Horus Media — reach that never blinks.",
    "Your podcast, heard by thousands. Sky-high ad placements start here.",
    "Grow your list, not just your followers. Newsletter strategy that converts.",
    "The truth is in the data. Anubis Analytics — measure what matters.",
    "Every decision, weighed with precision. Join 500+ brands we guide.",
    "Loyalty rewarded. Exclusive member insights, unlocked for you.",
    "Knowledge is power. Thoth Technology — tools built for curious minds.",
    "Your wisdom app, reimagined. Smarter workflows start with one download.",
    "Write the future of your business. Thoth's creator fund is now open.",
    "Your skin deserves ritual. Bastet Beauty — nine steps to radiant glow.",
    "Sacred ingredients, modern formulas. The cat-approved skincare routine.",
    "Moonlit, minimal, magical. The new ritual kit is here — limited stock.",
    "Taste the divine. Hathor Hospitality — where every meal is a ceremony.",
    "Book your golden getaway. Curated stays inspired by ancient celebration.",
    "Celebrate every season. Festival packages crafted for unforgettable moments.",
    "Shape something real. Khnum Craft — artisan kits for the creator in you.",
    "From the Nile to your studio. Clay, tools, and tradition in one bundle.",
    "Train like a warrior. Sekhmet Sport — built for those who refuse to quit.",
    "The desert doesn't wait. Sign up for the Lioness Run Series today.",
]


def _channel_multipliers(channel: str) -> dict:
    """Rough CPM/CTR/CVR profile per channel."""
    if channel == "Google Ads":
        return {"cpm": 130, "ctr": 0.048, "cvr": 0.065, "roas_base": 3.6}
    elif channel == "Meta Ads":
        return {"cpm": 75, "ctr": 0.022, "cvr": 0.038, "roas_base": 2.6}
    else:  # TikTok
        return {"cpm": 50, "ctr": 0.020, "cvr": 0.028, "roas_base": 2.1}


def generate_dataset() -> pd.DataFrame:
    rows = []
    start_date = date(2025, 9, 1)

    for client in CLIENTS:
        campaigns = CAMPAIGNS[client]
        for campaign in campaigns:
            n_channels = random.randint(2, 3)
            channels = random.sample(CHANNELS, n_channels)
            n_weeks = random.randint(5, 9)
            goal = random.choice(GOALS)
            audience = random.choice(AUDIENCES)
            ad_text = random.choice(AD_TEXTS)

            for channel in channels:
                m = _channel_multipliers(channel)
                for week_num in range(1, n_weeks + 1):
                    week_date = start_date + timedelta(weeks=week_num - 1)

                    base_spend = random.uniform(600, 6000)
                    ramp = 1 + 0.04 * (week_num - 1)
                    noise = random.uniform(0.75, 1.25)
                    spend = round(base_spend * ramp * noise, 2)

                    impressions = int(spend / m["cpm"] * 1000 * random.uniform(0.88, 1.12))
                    clicks = int(impressions * m["ctr"] * random.uniform(0.80, 1.20))
                    conversions = int(clicks * m["cvr"] * random.uniform(0.65, 1.35))
                    revenue = round(conversions * random.uniform(150, 1400) * random.uniform(0.88, 1.12), 2)

                    roas = round(revenue / spend, 2) if spend > 0 else 0.0
                    ctr  = round(clicks / impressions * 100, 2) if impressions > 0 else 0.0

                    rows.append({
                        "client":     client,
                        "campaign":   campaign,
                        "channel":    channel,
                        "week":       week_num,
                        "week_date":  week_date.strftime("%Y-%m-%d"),
                        "spend":      spend,
                        "impressions": impressions,
                        "clicks":     clicks,
                        "conversions": conversions,
                        "revenue":    revenue,
                        "roas":       roas,
                        "ctr":        ctr,
                        "ad_text":    ad_text,
                        "audience":   audience,
                        "goal":       goal,
                    })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    print("\nClients:", df["client"].unique())
    print("Channels:", df["channel"].unique())
