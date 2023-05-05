import unittest

from codename_ai.model.pmi.wiki_pmi import CalculateWordDistanceWithWikiPMI


class TestCalculateWordDistanceWithWikiPMI(unittest.TestCase):

    def test_run_with_freq(self):
        kyoki = {'A': {'A': 100, 'B': 10}}
        freq = {'A': 100, 'B': 20}
        target_word = 'A'
        doc_size = 10000

        resulted = CalculateWordDistanceWithWikiPMI._run(kyoki=kyoki, freq=freq, target_word=target_word, doc_size=doc_size, pmi_score_cutoff=-9999)
        expected = {'B': -3.912023005428146}
        self.assertAlmostEqual(resulted['B'], expected['B'], places=6)
        self.assertEqual(resulted.keys(), expected.keys())

    def test_run_without_freq(self):
        kyoki = {'A': {'A': 100, 'B': 10}}
        freq = {'A': 100}
        target_word = 'A'
        doc_size = 10000

        resulted = CalculateWordDistanceWithWikiPMI._run(kyoki=kyoki, freq=freq, target_word=target_word, doc_size=doc_size, pmi_score_cutoff=-9999)
        expected = {'B': 2.3025850929940455}
        self.assertAlmostEqual(resulted['B'], expected['B'], places=6)
        self.assertEqual(resulted.keys(), expected.keys())

    def test_run_without_any_data(self):
        kyoki = {'A': {'A': 100, 'B': 10}, 'C': {}}
        freq = {'A': 100}
        target_word = 'C'
        doc_size = 10000

        resulted = CalculateWordDistanceWithWikiPMI._run(kyoki=kyoki, freq=freq, target_word=target_word, doc_size=doc_size, pmi_score_cutoff=-9999)
        expected = {}
        self.assertEqual(resulted.keys(), expected.keys())

    def test_run_with_cutoff(self):
        kyoki = {'A': {'A': 100, 'B': 10, 'C': 2}}
        freq = {'A': 100, 'B': 20, 'C': 100}
        target_word = 'A'
        doc_size = 10000

        resulted = CalculateWordDistanceWithWikiPMI._run(kyoki=kyoki, freq=freq, target_word=target_word, doc_size=doc_size, pmi_score_cutoff=3)
        expected = {'B': -3.912023005428146}
        # CはPMI=0.69ぐらいなので消える
        self.assertAlmostEqual(resulted['B'], expected['B'], places=6)
        self.assertEqual(resulted.keys(), expected.keys())
