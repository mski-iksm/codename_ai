import gokart


class CalculateSynsetDistance(gokart.TaskOnKart):
    target_synset: str = luigi.Parameter()
    traverse_depth: int = luigi.IntParameter()

    def requires(self):
        synsets = _get_connected_synsets(synset=self.target_synset)
        return [CalculateSynsetDistance(target_synset=synset, traverse_depth=self.traverse_depth - 1) for synset in synsets] if self.traverse_depth >= 1 else []
