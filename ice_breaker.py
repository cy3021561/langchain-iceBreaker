from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

load_dotenv()

if __name__ == "__main__":
    print("Hello Langchain!")

    linkedin_url = linkedin_lookup_agent(name="Tom Yang USC Computer Science Machine Learning Taiwan")

    summary_template = """
        Given the Linkedin information {information} about a person from I want you to create:
        1. A short summary
        2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_url)

    print(chain.run(information=linkedin_data))
