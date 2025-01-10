import { TavilySearchResults } from "@langchain/community/tools/tavily_search"; // to create a new instance of the agent tool and  search the internet
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents"; // to create and execute an agent
import { createInterface } from "readline"; // to read user input from the terminal
import { AIMessage, HumanMessage } from "@langchain/core/messages"; // to create AI and Human language schema and store in history
import * as dotenv from "dotenv";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrieverTool } from "langchain/tools/retriever";
dotenv.config();

const model = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant called Max."],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

// Create and Assign tools
// 1. Custom data source - reading content from a url
const loader = new CheerioWebBaseLoader("https://js.langchain.com/docs/how_to");
const docs = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20,
});
const splitDocs = await splitter.splitDocuments(docs);
const embeddings = new OpenAIEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);
const retriever = vectorStore.asRetriever({
  k: 2,
});
const retrievalTool = createRetrieverTool(retriever, {
  name: "lcel_search",
  description:
    "Use this tool when searching for information about Langchain Expression Language (LCEl)",
});

// 2. Tavily Search Tool - leverage the openAI training
const searchTool = new TavilySearchResults();
const tools = [searchTool, retrievalTool];

// Create Agent
const agent = await createOpenAIFunctionsAgent({
  llm: model,
  prompt,
  tools,
});

// Create Agent Executor
const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

//Get user input
const rl = createInterface({
  input: process.stdin,
  output: process.stdout,
});

const chatHistory = [];

const askQuestion = () => {
  rl.question("User:", async (input) => {
    if (input.toLocaleLowerCase() === "exit") {
      rl.close();
      return;
    }

    // Call Agent
    const response = await agentExecutor.invoke({
      input: input,
      chat_history: chatHistory,
    });

    console.log("Agent: ", response.output);

    chatHistory.push(new HumanMessage(input));
    chatHistory.push(new AIMessage(response.output));

    askQuestion();
  });
};

askQuestion();
