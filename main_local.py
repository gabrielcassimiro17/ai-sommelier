from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import Pinecone
import langchain
import pinecone
import os
import streamlit as st

from chains import build_query_chain, build_recommendation_chain

langchain.debug = True
llm = GooglePalm(temperature=0.0)

pinecone.init(api_key=os.getenv("PINECONE_CLIENT_API"), environment="gcp-starter")
embeddings = GooglePalmEmbeddings()
vectordb = Pinecone.from_existing_index(index_name="wine-db", embedding=embeddings)


def search_wines(query, vectordb):
    docs = vectordb.similarity_search(query)
    return docs


def main():
    st.title("Wine Sommelier")

    st.write("Fill in the form in the sidebar to get a wine recommendation")

    # Ask the questions and store answers
    st.sidebar.subheader("1. Preferred Taste Profile")
    taste = st.sidebar.selectbox(
        "", ["Select an option", "Very sweet", "Sweet", "Neutral", "Dry", "Very dry"]
    )

    st.sidebar.subheader("2. Wine Experience")
    experience = st.sidebar.selectbox(
        "",
        [
            "Select an option",
            "Novice (just starting)",
            "Casual drinker (have tried a few)",
            "Enthusiast (drink regularly)",
            "Connoisseur (very knowledgeable)",
        ],
    )

    st.sidebar.subheader("3. Red vs. White")
    wine_color = st.sidebar.selectbox(
        "",
        [
            "Select an option",
            "Prefer red",
            "Prefer white",
            "Like both equally",
            "No preference",
        ],
    )

    st.sidebar.subheader("4. Favorite Flavors")
    flavors = [
        "Fruity",
        "Earthy",
        "Floral",
        "Spicy",
        "Oaky/Woody",
        "Citrusy",
        "Buttery",
        "Herbal",
    ]
    flavor = st.sidebar.multiselect("", flavors)

    st.sidebar.subheader("5. Pairing Intent")
    pairing = st.sidebar.selectbox(
        "",
        [
            "Select an option",
            "Casual drinking",
            "Romantic dinner",
            "Seafood meal",
            "Red meat dish",
            "Poultry dish",
            "Vegetarian meal",
            "Dessert",
            "No specific pairing",
        ],
    )

    st.sidebar.subheader(
        "6. Any complement of the answers above with other foods or tastes you like the most."
    )
    complement = st.sidebar.text_input("Complement the answers above")

    query_chain, query_response_format, query_output_parser = build_query_chain(llm)
    (
        recommend_chain,
        recommend_response_format,
        recommend_output_parser,
    ) = build_recommendation_chain(llm)

    if st.checkbox("Generate recommendation"):
        response = query_chain.run(
            {
                "taste": taste,
                "experience": experience,
                "wine_color": wine_color,
                "flavor": flavor,
                "pairing": pairing,
                "complement": complement,
                "response_format": query_response_format,
            }
        )

        query = query_output_parser.parse(response)["query_string"]
        print(query)
        docs = search_wines(query=query, vectordb=vectordb)

        wine_options = [
            {
                "name": doc.metadata["name"],
                "country": doc.metadata["country"],
                "province": doc.metadata["province"],
                "variety": doc.metadata["variety"],
                "winery": doc.metadata["winery"],
            }
            for doc in docs
        ]

        wine_1 = wine_options[0]
        wine_2 = wine_options[1]
        wine_3 = wine_options[2]

        response = recommend_chain.run(
            {
                "taste": taste,
                "experience": experience,
                "wine_color": wine_color,
                "flavor": flavor,
                "pairing": pairing,
                "complement": complement,
                "response_format": recommend_response_format,
                "wine_1": wine_1,
                "wine_2": wine_2,
                "wine_3": wine_3,
            }
        )

        recommendation = recommend_output_parser.parse(response)["recommendation"]
        explanation = recommend_output_parser.parse(response)["explanation"]

        matching_wine = next(
            (wine for wine in wine_options if wine["name"] == recommendation), None
        )

        st.write("---")
        try:
            st.write(
                f"""
                    ### 🍷 **Recommended Wine**
                    {recommendation}

                    ---
                    **Country**: {matching_wine['country']}

                    **Province**: {matching_wine['province']}

                    **Variety**: {matching_wine['variety']}

                    **Winery**: {matching_wine['winery']}

                    **Explanation**: {explanation}
                    """
            )
        except:
            st.write(
                f"""
                    ### 🍷 **Recommended Wine**
                    {recommendation}

                    **Explanation**: {explanation}
                    """
            )


if __name__ == "__main__":
    main()
