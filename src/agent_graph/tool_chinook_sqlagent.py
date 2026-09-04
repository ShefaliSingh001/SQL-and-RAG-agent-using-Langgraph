from typing import List
from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.tools import tool

import sys
from pyprojroot import here
root_dir = str(here())
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
from src.agent_graph.load_tools_config import LoadToolsConfig

TOOLS_CFG = LoadToolsConfig()


class Table(BaseModel):
    """
    Represents a table in the SQL database.

    Attributes:
        name (str): The name of the table in the SQL database.
    """

    name: str = Field(description="Name of table in SQL database.")


class TableList(BaseModel):
    tables: List[Table]


def get_tables(categories: List[Table]) -> List[str]:
    """Maps category names to corresponding SQL table names.

    Args:
        categories (List[Table]): A list of `Table` objects representing different categories.

    Returns:
        List[str]: A list of SQL table names corresponding to the provided categories.
    """
    tables = []
    for category in categories:
        if category.name == "Music":
            tables.extend(
                [
                    "Album",
                    "Artist",
                    "Genre",
                    "MediaType",
                    "Playlist",
                    "PlaylistTrack",
                    "Track",
                ]
            )
        elif category.name == "Business":
            tables.extend(
                ["Customer", "Employee", "Invoice", "InvoiceLine"])
    return tables


class ChinookSQLAgent:

    def __init__(
        self,
        sqldb_directory: str,
        llm: str,
        llm_temerature: float
    ) -> None:

        self.sql_agent_llm = ChatGroq(
            model=llm,
            temperature=llm_temerature
        )

        self.db = SQLDatabase.from_uri(
            f"sqlite:///{sqldb_directory}"
        )

        print(self.db.get_usable_table_names())

        category_chain_system = """
        Determine which category is relevant to the user question.

        The available categories are:

        Music
        Business

        Return all relevant categories.
        """

        category_chain = (
            ChatPromptTemplate.from_messages([
                ("system", category_chain_system),
                ("human", "{input}")
            ])
            | self.sql_agent_llm.with_structured_output(TableList)
        )

        table_chain = (
            category_chain
            | (lambda x: get_tables(x.tables))
        )

        query_chain = create_sql_query_chain(
            self.sql_agent_llm,
            self.db
        )

        table_chain = (
            {"input": itemgetter("question")}
            | table_chain
        )

        self.full_chain = (
            RunnablePassthrough.assign(
                table_names_to_use=table_chain
            )
            | query_chain
        )


@tool
def query_chinook_sqldb(query: str) -> str:
    """Query the Chinook SQL Database. Input should be a search query."""
    agent = ChinookSQLAgent(
        sqldb_directory=TOOLS_CFG.chinook_sqldb_directory,
        llm=TOOLS_CFG.chinook_sqlagent_llm,
        llm_temerature=TOOLS_CFG.chinook_sqlagent_llm_temperature
    )

    query = agent.full_chain.invoke({"question": query})

    return agent.db.run(query)
