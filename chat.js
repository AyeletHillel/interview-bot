import { config } from "dotenv";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
config();

async function initializeChain() {
    const textLoader = fs.readFileSync("interview.txt", "utf8");

    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
    const docs = await textSplitter.createDocuments([textLoader]);

    const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

    const model = new ChatOpenAI({
        modelName: "gpt-3.5-turbo",
    });

    const chain = ConversationalRetrievalQAChain.fromLLM(
        model,
        vectorStore.asRetriever(),     
        {
            returnSourceDocuments: false,
            questionGeneratorChainOptions: {
                llm: model,
                template: "Q: {question} A: {answer}",
            },
            verbose: true,
        }
    );

    return chain;
}

const chainPromise = initializeChain();

export default chainPromise;
