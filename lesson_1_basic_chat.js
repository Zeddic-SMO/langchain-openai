import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";

dotenv.config();

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
});

const response = await model.invoke("What is the meaning of life?");

console.log(response.content);
