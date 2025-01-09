// The model will be able to recall previous history and the user will be able to ask follow-up questions

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"; //to load web pages and read contents
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"; //to create chains that combine documents
import { createRetrievalChain } from "langchain/chains/retrieval"; // to create a retrieval chain
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"; //to split text into smaller chunks
import { MemoryVectorStore } from "langchain/vectorstores/memory"; // to store vectors in memory
import { AIMessage, HumanMessage } from "@langchain/core/messages"; // to create messages - mimicing a chat history
import { MessagesPlaceholder } from "@langchain/core/prompts"; // to create a placeholder for messages
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever"; //to create a retriever that is aware of the history

import * as dotenv from "dotenv";
dotenv.config();

// Load data and create vector store
const createVectorStore = async () => {
  // reading content from a url
  const loader = new CheerioWebBaseLoader(
    "https://js.langchain.com/docs/how_to"
  );

  const documentC = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20,
  });

  const splitDocs = await splitter.splitDocuments(documentC);
  // console.log(splitDocs);

  // We need to get the most relevant doc - hence we need a vector store: we need to convert the data in a format the vector store can understand
  const embeddings = new OpenAIEmbeddings();

  // we will use in memory store but this data can be store permanently in a production application
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  return vectorStore;
};

// Create Retrieval chain
const createChain = async () => {
  // Instantiate the model
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.9,
  });

  // Create Prompt Template from fromMessages
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's questions based on the following context: {context}.",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);

  // Create the Chain
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,
  });

  // Retrieve data (THIS RETRIEVAL DOES NOT ALLOW US TO PASS IN THE CHAT HISTORY HENCE THE OPTIMIZATION BELOW)
  const retriever = vectorStore.asRetriever({
    k: 2,
  });

  // Optimization of retrieval with historyawareretrieval - that takes the chat history along the user's input as query
  // Taking into account the users chat history as well as the data fetched from the webpage
  const retrievalPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);

  const historyAwareRetrieval = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: retrievalPrompt,
  });

  // Tie everything with a retriever chain
  const conversationChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: historyAwareRetrieval,
  });

  return conversationChain;
};

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);

// Chat History
const chatHistory = [
  new HumanMessage("Hello"),
  new AIMessage("HI, how can I help you"),
  new HumanMessage("My name is John"),
  new AIMessage("Nice to meet you John"),
  new HumanMessage("What is LCEL"),
  new AIMessage("LCEL stands for Langchain Expression Language"),
];

const response = await chain.invoke({
  input: "what my name?", //without the history the model will not what is it we're referring to.
  chat_history: chatHistory,
});

console.log(response.answer);
