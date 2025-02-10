from llama_cpp import Llama
from pathlib import Path
import os
import logging 


BASE_PATH = Path(__file__).parent.absolute()
os.makedirs(Path(BASE_PATH).joinpath("log"), exist_ok=True)
logging.basicConfig(filemode="w", filename="log/logs.log", level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


class LlamaModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):

        logging.info("Loading the Model")
        try:
            self.model = Llama(
                model_path=self.model_path, verbose=False
            )

        except Exception as E:
            logging.error(f"Error loading the model: {E}")
            return None

        else:
            logging.info("Model Loaded Successfully")
            return self.model


    def inference(self, input, history):
        try:
            logging.info("Generating Answer from model")
            result = self.model.create_chat_completion(
                messages=[
                    {"role" : "system", "content" : "You are an assistant who has diverse knowlegde in Machine learning and AI. Your task is to provide solutions to user query in 3-4 lines. Ensure your length of your answer is not more than 256 tokens."},
                    {"role": "user", "content": input},
                ],
                max_tokens=256,
                # stream=True
            )
            print(result)

        except Exception as E:
            logging.error(f"Error during inference: {E}")

        else:
            logging.info("ANswer Generated form model")
            # s = ""
            # for i in result:
            #     # s += i
            #     print(i)
            #     yield i
            history.append([input, result.get("choices")[0].get("message").get("content")])
            # history.append([input, s])
            print(history)
            return history[-1][1]