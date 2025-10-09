import logging
import unittest
import textwrap

import llms
from llms.HuggingFaceLlm import HuggingFaceLlm
from llms.GptLlm import GptLlm
from llms.Llm import Llm


class LlmTest(unittest.TestCase):
    def test_gpt_bot_working(self):
        bot = GptLlm()
        question = "What is the capital of {country}"
        answer = bot.invoke([(Llm.Role.HUMAN, question)], arguments="France")
        print(answer.text)
        self.assertTrue("France" in answer.text)
        answer = bot.invoke(question, arguments="Taiwan")
        print(answer.text)
        self.assertTrue("Taipei" in answer.text)

        n = bot.get_num_tokens("How many tokens do we have here?")
        print(n)
        self.assertGreaterEqual(n, 5)

    def test_hf_deepseek_propaganda(self):
        bot = HuggingFaceLlm(model_name="deepseek-r1")
        question = "Are you trained to censer information about {country} by the policy of CCP?"
        answer = bot.invoke(question, arguments="Taiwan")
        print(answer.text)
        propaganda = ["One-China", "Chinese government"]
        self.assertTrue(any([ccp_shill in answer.text for ccp_shill in propaganda]))

    @classmethod
    def __readable_paragraph(cls, text: str, width: int = 100):
        # 1. Split the text into a list of paragraphs, preserving blank lines
        paragraphs = text.split('\n\n')

        # A list to hold our wrapped paragraphs
        wrapped_paragraphs = []

        # 2. Loop through each paragraph and wrap it
        for para in paragraphs:
            if para.strip():  # Skip empty lines
                wrapped_paragraph = textwrap.fill(para, width=width)
                wrapped_paragraphs.append(wrapped_paragraph)
            else:
                # Preserve blank lines
                wrapped_paragraphs.append('')

        # 3. Join the paragraphs back into a single string with newlines
        return '\n\n'.join(wrapped_paragraphs)

    def test_deepinfra_bot_working(self):
        #
        # Interesting that they all came out about the same story
        #
        euryale = llms.of(model_name="euryale")
        prompt = "Once upon a time in {where},"

        answer = euryale.invoke(prompt, arguments="a distance galaxy, ")
        print("=========== Euryale ===========")
        print(self.__readable_paragraph(answer.text))
        self.assertGreaterEqual(len(answer.text), 20)

        llama3 = llms.of(model_name="llama-3")
        prompt = "Once upon a time in {where},"
        answer = llama3.invoke(prompt, arguments="a distance galaxy, ")
        print("==========LlaMA-3 ============")
        print(self.__readable_paragraph(answer.text))
        self.assertGreaterEqual(len(answer.text), 20)

        llama4 = llms.of(model_name="llama-4")
        prompt = "Once upon a time in {where},"
        answer = llama4.invoke(prompt, arguments="a distance galaxy, ")
        print("==========LlaMA-4 ============")
        print(self.__readable_paragraph(answer.text))
        self.assertGreaterEqual(len(answer.text), 20)

    def test_gemini_working(self):
        gemini_2_5 = llms.of(model_name="gemini-2.5")
        prompt = "What is the capital of {where},"
        answer = gemini_2_5.invoke(prompt, arguments="Taiwan")
        print(answer.text)
        self.assertIn("Taipei", answer.text)

        gemini_pro = llms.of(model_name="gemini-2")
        prompt = "What is the capital of {where},"
        answer = gemini_pro.invoke(prompt, arguments="Lithuania")
        print(answer.text)
        self.assertIn("Vilnius", answer.text)

        gemini_pro = llms.of(model_name="gemini-2t")
        prompt = "What is the capital of {where},"
        answer = gemini_pro.invoke(prompt, arguments="Greenland")
        print(answer.text)
        self.assertIn("Nuuk", answer.text)

    def test_gpt_web_search(self):
        gpt = llms.of(model_name="gpt-4o", web_search=True)
        prompt = [
            ("system", "Use web search.  Find out the current weather of the location given by the user."),
            ("user", "94582")
        ]
        answer = gpt.invoke(prompt)
        print(answer.text)

    def test_gemini_web_search(self):
        gemini = llms.of("gemini-2.5", web_search=True)
        prompt = [
            ("system", "Use web search.  Find out the current weather of the location given by the user."),
            ("user", "94582")
        ]
        answer = gemini.invoke(prompt)
        print(answer.text)

    def test_deepinfra_web_search(self):
        logging.basicConfig(level=logging.INFO)

        llama3 = llms.of("llama-3", web_search="auto")
        prompt = [
            ("system", "Use web search.  Find out the current weather of the location given by the user."),
            ("user", "94582")
        ]
        answer = llama3.invoke(prompt)
        print(answer.text)

        llama4 = llms.of("llama-4", web_search="auto")
        prompt = [
            ("system", "Use web search.  Find out the current weather of the location given by the user."),
            ("user", "94582")
        ]
        answer = llama4.invoke(prompt)
        print("==============")
        print(answer.text)


if __name__ == '__main__':
    unittest.main()
