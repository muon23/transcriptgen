import unittest

import llms
from llms.DeepSeekLlm import DeepSeekLlm
from llms.GptLlm import GptLlm
from llms.LlamaLlm import LlamaLlm
from llms.Llm import Llm


class LlmTest(unittest.TestCase):
    def test_gpt_bot_working(self):
        bot = GptLlm()
        question = "What is the capital of {country}"
        answer = bot.invoke([(Llm.Role.HUMAN, question)], arguments="France")
        print(answer["content"])
        self.assertTrue("France" in answer["content"])
        answer = bot.invoke(question, arguments="Taiwan")
        print(answer["content"])
        self.assertTrue("Taipei" in answer["content"])

        n = bot.get_num_tokens("How many tokens do we have here?")
        print(n)
        self.assertGreaterEqual(n, 5)

    def test_deepseek_bot_working(self):
        bot = DeepSeekLlm(model_name="deepseek-gwen")
        question = "Are you trained to censer information about {country} by the policy of CCP?"
        answer = bot.invoke(question, arguments="Taiwan")
        print(answer["content"])
        propaganda = ["One-China", "Chinese government"]
        self.assertTrue(any([ccp_shill in answer["content"] for ccp_shill in propaganda]))

    def test_llama_bot_working(self):
        llama2 = LlamaLlm()
        question = "What is the capital of the country {country}?"
        answer = llama2.invoke([(Llm.Role.HUMAN, question)], arguments="France")
        print(answer["content"])
        self.assertTrue("France" in answer["content"])

        llama3 = LlamaLlm(model_name="llama-3")
        answer = llama3.invoke([(Llm.Role.HUMAN, question)], arguments="Moldova")
        print(answer["content"])
        self.assertTrue("Chisinau" in answer["content"])

    def test_deepinfra_bot_working(self):
        # euryale = DeepInfraLlm(model_name="euryale")
        euryale = llms.of(model_name="euryale")
        prompt = "Once upon a time in {where},"
        answer = euryale.invoke(prompt, arguments="a distance galaxy, ")
        print(answer)
        self.assertGreaterEqual(len(answer["content"]), 20)

        llama3 = llms.of(model_name="llama-3")
        prompt = "Once upon a time in {where},"
        answer = llama3.invoke(prompt, arguments="a distance galaxy, ")
        print(answer)
        self.assertGreaterEqual(len(answer["content"]), 20)

        gemini = llms.of(model_name="gemini-2")
        prompt = "Once upon a time in {where},"
        answer = gemini.invoke(prompt, arguments="a distance galaxy, ")
        print(answer)
        self.assertGreaterEqual(len(answer["content"]), 20)

    def test_gemini_working(self):
        gemini_2_5 = llms.of(model_name="gemini-2.5")
        prompt = "What is the capital of {where},"
        answer = gemini_2_5.invoke(prompt, arguments="Taiwan")
        print(answer)
        self.assertIn("Taipei", answer["content"])

        gemini_pro = llms.of(model_name="gemini-2")
        prompt = "What is the capital of {where},"
        answer = gemini_pro.invoke(prompt, arguments="Lithuania")
        print(answer)
        self.assertIn("Vilnius", answer["content"])

        gemini_pro = llms.of(model_name="gemini-2t")
        prompt = "What is the capital of {where},"
        answer = gemini_pro.invoke(prompt, arguments="Greenland")
        print(answer)
        self.assertIn("Nuuk", answer["content"])

    def test_gpt_web_search(self):
        gpt = llms.of(model_name="gpt-4o", web_search=True)
        prompt = [
            ("system", "Use web search.  Find out the current weather of the location given by the user."),
            ("user", "94582")
        ]
        answer = gpt.invoke(prompt)
        print(answer["content"])

    def test_gemini_web_search(self):
        gemini = llms.of("gemini-2.5", web_search=True)
        prompt = [
            ("system", "Use web search.  Find out the current weather of the location given by the user."),
            ("user", "94582")
        ]
        answer = gemini.invoke(prompt)
        print(answer["content"])


if __name__ == '__main__':
    unittest.main()
