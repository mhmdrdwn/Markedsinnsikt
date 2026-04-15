import pandas as pd
import random
from datetime import date, timedelta

_SEED = 7391  # project-specific seed — reproducible but unique

CLIENTS = [
    "Salah Brand",
    "Haaland Brand",
    "Mbappe Brand",
    "Vinicius Brand",
    "Bellingham Brand",
    "Rodri Brand",
    "Yamal Brand",
]

CAMPAIGNS = {
    "Salah Brand":      ["King of Egypt Kit", "Ramadan Community Drive", "Charity Foundation Push"],
    "Haaland Brand":    ["Goal Machine Collection", "Nordic Fitness Series", "Champions Energy Launch"],
    "Mbappe Brand":     ["Speed Icon Drop", "Real Deal Launch", "Youth Academy Fund"],
    "Vinicius Brand":   ["Vini Jr Street Culture", "Samba Energy Push", "Anti-Racism Stand"],
    "Bellingham Brand": ["Bellingham x Dior", "Real Debut Collection", "England Rising Star"],
    "Rodri Brand":      ["Ballon d'Or Celebration", "La Liga Masterclass", "Calm Under Pressure"],
    "Yamal Brand":      ["Next Gen Launch", "Barcelona Academy", "Gen Z Wave"],
}

CHANNELS = ["Google Ads", "Meta Ads", "TikTok Ads"]

GOALS = ["Brand Awareness", "Lead Generation", "Direct Sales", "App Installs"]

AUDIENCES = [
    "18-30 football fans and streetwear enthusiasts",
    "16-24 Gen Z sports and culture followers",
    "25-40 premium sports lifestyle shoppers",
    "18-28 aspiring athletes and fitness enthusiasts",
    "20-35 football gaming and fantasy sports players",
    "30-50 sports memorabilia and kit collectors",
    "16-22 social media natives following football culture",
    "25-45 brand-conscious football supporters",
    "18-32 charity and socially engaged sports fans",
    "22-38 multi-sport performance apparel shoppers",
]

AD_TEXTS = [
    "Mo Salah's official collection. Worn on the pitch. Built for the streets.",
    "Give back this Ramadan. Every purchase supports Mo Salah's Foundation.",
    "The King has a new kit. Limited edition — available while stocks last.",
    "Train like Haaland. The Nordic Fitness Series starts this week.",
    "Pure power. Pure focus. The Goal Machine Collection is here.",
    "Champions run on more than talent. Fuel your game with Haaland Energy.",
    "Mbappé moves at his own speed. The Speed Icon Drop — one night only.",
    "A new era begins. Mbappé x Real Madrid — the official launch collection.",
    "Invest in the next generation. Mbappé's Youth Academy Fund is open.",
    "Vini Jr brings the street to the pitch. The collab you didn't see coming.",
    "Samba never stops. Vinicius Jr's limited streetwear drop is live.",
    "Football is for everyone. Stand with Vini Jr — wear the message.",
    "Bellingham x Dior. Where football meets fashion. Exclusively yours.",
    "The debut that changed everything. Own a piece of the Real Madrid story.",
    "England's next captain. The Rising Star collection — inspired by Jude.",
    "Rodri. Ballon d'Or. One collection to mark a historic season.",
    "Read the game. Control the tempo. La Liga Masterclass — now available.",
    "Composure is a skill. Train with Rodri's performance series.",
    "Lamine Yamal is just getting started. The Next Gen Launch is here.",
    "Born in 2007. Barcelona's future. Yamal's debut collection — limited run.",
    "Gen Z runs football now. Join the wave — Yamal x TikTok collab drop.",
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
