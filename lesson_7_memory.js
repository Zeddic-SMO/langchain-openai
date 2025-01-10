import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import * as dotenv from "dotenv";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";
dotenv.config();

// model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

// prompt
const prompt = ChatPromptTemplate.fromTemplate(
  `
    You are an AI assistant.
    History: {history}
    {input}
    `
);

// redis db
const upstashChatHistory = new UpstashRedisChatMessageHistory({
  sessionId: "chat1", //use uuid to generate a unique id
  config: {
    url: process.env.UPSTASH_REDIS_URL,
    token: process.env.UPSTASH_REDIS_TOKEN,
  },
});

// Memory
const memory = new BufferMemory({
  memoryKey: "history",
  chatHistory: upstashChatHistory,
});

// Using the Chain Classes
const chain = new ConversationChain({
  llm: model,
  prompt: prompt,
  memory,
});

// Using LCEL
// const chain = prompt.pipe(model);

// Get response
// console.log("Initial History:", await memory.loadMemoryVariables());
// const input1 = {
//   input: "Hello, My name is Leon.",
// };
// const response1 = await chain.invoke(input1);
// console.log("Res1", response1);

console.log("Updated History:", await memory.loadMemoryVariables());
const input2 = {
  input: "What is my name?",
};
const response2 = await chain.invoke(input2);

console.log("Res1", response2);
