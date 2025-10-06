import unittest

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

    def test_deepinfra_bot_working(self):
        # euryale = DeepInfraLlm(model_name="euryale")
        euryale = llms.of(model_name="euryale")
        prompt = "Once upon a time in {where},"
        answer = euryale.invoke(prompt, arguments="a distance galaxy, ")
        print(answer)
        self.assertGreaterEqual(len(answer.text), 20)

        llama3 = llms.of(model_name="llama-4")
        prompt = "Once upon a time in {where},"
        answer = llama3.invoke(prompt, arguments="a distance galaxy, ")
        print(answer)
        self.assertGreaterEqual(len(answer.text), 20)

        gemini = llms.of(model_name="gemini-2")
        prompt = "Once upon a time in {where},"
        answer = gemini.invoke(prompt, arguments="a distance galaxy, ")
        print(answer)
        self.assertGreaterEqual(len(answer.text), 20)

    def test_gemini_working(self):
        gemini_2_5 = llms.of(model_name="gemini-2.5")
        prompt = "What is the capital of {where},"
        answer = gemini_2_5.invoke(prompt, arguments="Taiwan")
        print(answer)
        self.assertIn("Taipei", answer.text)

        gemini_pro = llms.of(model_name="gemini-2")
        prompt = "What is the capital of {where},"
        answer = gemini_pro.invoke(prompt, arguments="Lithuania")
        print(answer)
        self.assertIn("Vilnius", answer.text)

        gemini_pro = llms.of(model_name="gemini-2t")
        prompt = "What is the capital of {where},"
        answer = gemini_pro.invoke(prompt, arguments="Greenland")
        print(answer)
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


if __name__ == '__main__':
    unittest.main()
