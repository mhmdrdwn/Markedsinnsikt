import pandas as pd
import random
from datetime import date, timedelta

_SEED = 7391  # project-specific seed — reproducible but unique

CLIENTS = [
    "SalahStyle",
    "HaalandFit",
    "MbappeMode",
    "ViniJrVibe",
    "BellinghamBrand",
    "RodriSport",
    "YamalYouth",
]

CAMPAIGNS = {
    "SalahStyle":      ["Winter Capsule", "Outlet Weekend", "Influencer Drop"],
    "HaalandFit":      ["New Year Reset", "Spring Detox", "Marathon Prep"],
    "MbappeMode":      ["Streetwear Release", "Back to Basics", "Collab Drop"],
    "ViniJrVibe":      ["Podcast Ad Push", "Newsletter Growth", "Creator Fund"],
    "BellinghamBrand": ["Cabin Season", "Holiday Decor", "Summer Refresh"],
    "RodriSport":      ["Outdoor Season", "Winter Sports", "Gear Clearance"],
    "YamalYouth":      ["Loyalty Launch", "Weekend Flash", "Member Perks"],
}

CHANNELS = ["Google Ads", "Meta Ads", "TikTok Ads"]

GOALS = ["Brand Awareness", "Lead Generation", "Direct Sales", "App Installs"]

AUDIENCES = [
    "18-30 urban runners",
    "25-40 design-conscious homeowners",
    "22-35 streetwear enthusiasts",
    "30-50 outdoor adventurers",
    "18-28 health-conscious students",
    "35-55 premium lifestyle seekers",
    "20-32 social media natives",
    "28-45 remote workers",
    "40-60 renovation planners",
    "18-24 Gen Z trendsetters",
    "30-45 working parents",
    "50-65 active retirees",
]

AD_TEXTS = [
    "Reset your health in January. Science-backed programs for real results.",
    "Spring into shape — 6-week detox plans designed for active lifestyles.",
    "Train smarter, not harder. Official marathon prep starts now.",
    "Capsule wardrobe essentials, crafted for Nordic winters.",
    "Outlet prices on premium pieces. This weekend only.",
    "The drop everyone's been waiting for. Limited pieces, local talent.",
    "Streetwear built different. New release, limited run.",
    "Back to the essentials. Quality over quantity — always.",
    "Two names, one collection. The collab you didn't see coming.",
    "Your voice, amplified. Podcast advertising that actually converts.",
    "Grow your list, not just your followers. Newsletter strategy that works.",
    "Fund the creators who matter. Apply for the Creator Fund today.",
    "Your cabin, fully stocked. Essentials delivered before the season.",
    "Scandinavian holiday decor. Minimal, warm, and entirely yours.",
    "Refresh your space this summer. New arrivals, every week.",
    "Gear up for the outdoors. Expert picks for every trail.",
    "Winter sports, sorted. Skis, boots, and everything in between.",
    "Last season's best gear, this season's lowest prices.",
    "Loyalty has its rewards. Join and unlock exclusive member deals.",
    "Flash sale this weekend. Up to 40% off selected lines.",
    "Members get more. Exclusive perks, early access, zero fuss.",
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
    random.seed(_SEED)
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
