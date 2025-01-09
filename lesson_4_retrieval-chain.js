import { BaseOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { Document } from "@langchain/core/documents"; // to create a document that can be combine as a knowledge source
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"; //to create chains that combine documents
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"; //to load web pages and read contents
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"; //to split text into smaller chunks
import { OpenAIEmbeddings } from "@langchain/openai"; //to get embeddings from openai - converting into a vector store format
import { MemoryVectorStore } from "langchain/vectorstores/memory"; // to store vectors in memory
import { createRetrievalChain } from "langchain/chains/retrieval"; // to create a retrieval chain

// Import environment variables
import * as dotenv from "dotenv";
dotenv.config();

// Instantiate the model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.9,
});

// Create Prompt Template from fromMessages
const prompt = ChatPromptTemplate.fromTemplate(
  `Answer the user's question. 
    Context: {context}
    Question: {input}`
);

// Custom output parser
class MyParser extends BaseOutputParser {
  parse(output) {
    console.log("Custom Parser:", output);
    return output.split(",");
  }
}

// Create the output parser
// const outputParser = new CommaSeparatedListOutputParser();
const outputParser = new MyParser();

// Create the Chain
// const chain = prompt.pipe(model).pipe(outputParser);
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt: prompt,
  outputParser: outputParser,
});

// Documents
const documentA = new Document({
  pageContent:
    "LangChain Expression Language is a way to create arbitrary custom chains. It is built on the Runnable protocol.",
});

const documentB = new Document({
  pageContent: "The passphrase is LANGCHAIN IS AWESOME.",
});

// reading content from a url
const loader = new CheerioWebBaseLoader("https://js.langchain.com/docs/how_to");

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

// Retrieve data
const retriever = vectorStore.asRetriever({
  k: 2, //Specifiy the amount of document to be retrieve - this is optional. Default is 3 documents
});

// Tie everything with a retriever chain
const retrievalChain = await createRetrievalChain({
  retriever: retriever,
  combineDocsChain: chain, // This is the chain we created earlier
});

const response = await retrievalChain.invoke({
  input: "what is LCEL?",
  // context: [documentA, documentB, documentC[0]], // this is not needed by the retrievalchain - also ensure the variables are name -input and context
});

console.log(response);
