
import os
import inquirer

from dotenv import load_dotenv
import shutup; shutup.please()

from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from colorama import Fore

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY'].strip()


def init():

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo-16k",
        temperature=0.2
    )

    inq = inquirer.prompt([
        inquirer.Text("youtube_url", message=Fore.GREEN +
                      "Please enter video url")
    ])

    # Retrieve
    docs = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    ).split_documents(
        YoutubeLoader.from_youtube_url(
            inq['youtube_url'],
            add_video_info=False,
        ).load()
    )
    # Summarize
    res = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce"
    ).run(docs)

    print(Fore.RED + f"\n~~~~ Summary ~~~~\n")
    print(Fore.BLUE + f"{res}\n")

    inq = inquirer.prompt([
        inquirer.List(
            "task",
            message=Fore.GREEN + "What do you want to do?",
            choices=["Ask this video",
                     "Restart",
                     "Exit"
                     ],
            default=["Ask this video"]
        ),
    ])

    # Ask The Video
    if inq['task'] == 'Ask this video':
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=FAISS.from_documents(
                docs, OpenAIEmbeddings()).as_retriever()
        )

        while True:
            query = input(
                Fore.GREEN + "What is your question? ('q' enter to exit)\n")
            if query == "q":
                break

            res = qa_chain.run(query)
            print(Fore.BLUE + f"\n\nAnswer: {res}\n\n")

    elif inq['task'] == 'Restart':
        init()


if __name__ == '__main__':
    print(Fore.RED + f" ~~~ Youtube Video Indexer ~~~ \n")
    init()
