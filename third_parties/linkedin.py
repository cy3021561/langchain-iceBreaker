import os
import requests


def scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    headers = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}

    response = requests.get(
        api_endpoint,
        params={"linkedin_profile_url": linkedin_profile_url},
        headers=headers,
    )
    # response = requests.get(
    #     "https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"
    # )
    data = response.json()
    data = {
        key: val
        for key, val in data.items()
        if val not in ([], "", "", None)
        and key not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data
