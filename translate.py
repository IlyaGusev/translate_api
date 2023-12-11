import ctranslate2
import sentencepiece as spm

from tqdm import tqdm
from nltk import sent_tokenize


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


class Translator:
    def __init__(self, sp_model_path, ct_model_path, max_segment_length: int = 300):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.translator = ctranslate2.Translator(ct_model_path, "cuda")
        self.max_segment_length = max_segment_length

    def translate(
        self,
        text,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "rus_Cyrl"
    ):
        text = text.strip()
        entries = self.segment_text(text, 0)
        entries = self._translate_segments(entries, src_lang, tgt_lang)
        return self._restore_targets(entries)[0]

    def translate_records(
        self,
        records,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "rus_Cyrl"
    ):
        for batch in tqdm(list(gen_batch(records, batch_size=512))):
            entries = []
            for record in batch:
                text_id = record["text_id"]
                text = record["text"].strip()
                entries.extend(self.segment_text(text, text_id))

            entries = self._translate_segments(entries, src_lang, tgt_lang)
            target_texts = self._restore_targets(entries)
            assert len(batch) == len(target_texts), str(batch) + "\n\n" + str(target_texts)
            for record, target_text in zip(batch, target_texts):
                record["translation"] = target_text
        return records

    def segment_text(self, text, text_id):
        segments = []
        if "\n" in text:
            fragments = [f for f in text.split("\n") if f.strip()]
            for fragment in fragments:
                if len(fragment) < self.max_segment_length:
                    segments.append({"sentence": fragment, "text_id": text_id, "delimiter": "\n"})
                else:
                    for sentence in sent_tokenize(fragment):
                        segments.append({"sentence": sentence, "text_id": text_id, "delimiter": " "})
                segments[-1]["delimiter"] = "\n"
        else:
            if len(text) < self.max_segment_length:
                segments.append({"sentence": text, "text_id": text_id, "delimiter": ""})
            else:
                for sentence in sent_tokenize(text):
                    segments.append({"sentence": sentence, "text_id": text_id, "delimiter": " "})
        return segments

    def _translate_segments(self, entries, src_lang, tgt_lang):
        target_prefix = [[tgt_lang]] * len(entries)
        source_sents_subworded = self.sp.encode_as_pieces([r["sentence"] for r in entries])
        source_sents_subworded = [[src_lang] + sent + ["</s>"] for sent in source_sents_subworded]

        translations_subworded = self.translator.translate_batch(
            source_sents_subworded,
            batch_type="tokens",
            max_batch_size=4096,
            beam_size=5,
            target_prefix=target_prefix
        )
        translations_subworded = [translation.hypotheses[0] for translation in translations_subworded]
        for translation in translations_subworded:
            if tgt_lang in translation:
                translation.remove(tgt_lang)

        translations = self.sp.decode(translations_subworded)
        for entry, target in zip(entries, translations):
            if entry["sentence"]:
                entry["target"] = target
            else:
                entry["target"] = ""
        return entries

    def _restore_targets(self, entries):
        target_texts = []
        current_sentences = []
        current_text_id = None
        for entry in entries:
            text_id = entry["text_id"]
            if current_text_id is not None and text_id != current_text_id:
                target_texts.append("".join(current_sentences).strip())
                current_sentences = []
            current_text_id = text_id
            current_sentences.append(entry["target"])
            current_sentences.append(entry["delimiter"])
        target_texts.append(" ".join(current_sentences))
        return target_texts
