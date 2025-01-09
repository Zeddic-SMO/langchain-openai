import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as dotenv from "dotenv";

dotenv.config();

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

// Use prompts to controll the chat and narrow it to a specific user input

//1.  Create Prompt template
// const prompt = ChatPromptTemplate.fromTemplate(
//   "You are a comedian. Tell a joke on the following word {input}"
// );

// console.log(await prompt.format({ input: "chicken" }));

//2.  Create Prompt Template from fromMessages
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are a talented chef.  Create a recipe based on a main ingredient provided by the user.",
  ],
  ["human", "{input}"],
]);

// Create chain - by creating a variable
const chain = prompt.pipe(model);

// call chain
const res = await chain.invoke({
  input: "dog",
});

console.log(res);
