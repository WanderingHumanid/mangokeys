package com.mangokeyboard.transliterator

import java.io.File

/**
 * The keyboard on the phone.
 *
 * The model says which word the user probably meant, the word list says
 * which words exist, and the language model says which of them is the
 * natural one. There are two paths.
 *
 * `quick` uses the rules and the word list only. It takes well under a
 * millisecond, so it can run on every keystroke.
 *
 * `suggest` adds the models. It takes a few milliseconds, so it runs when the
 * typing pauses.
 *
 * There are two models. The transliteration model was trained on whole words
 * and is the better of the two at them. The prefix model was trained on the
 * same words plus the pairs where the roman side is cut short, so it learns
 * to stop in the middle of a word and the word list completes it. Neither is
 * good at the other's job, and both are six megabytes.
 */
class Keyboard(
    onnxDirectory: File,
    tokenizerJson: String,
    private val lexicon: Lexicon,
    private val ngram: CharNgram? = null,
    prefixModel: File? = null,
    beam: Int = 4,
    private val limit: Int = 8,
) {
    private val transliterator = Transliterator(
        onnxDirectory, tokenizerJson, maxSrc = 48, maxTgt = 64, beam = beam,
    )
    private val prefixModel = prefixModel?.let {
        Transliterator(it, tokenizerJson, maxSrc = 48, maxTgt = 64, beam = 1)
    }
    private val engine = SuggestionEngine(lexicon, ngram)

    /** The Malayalam of a word, without the word list. */
    fun transliterate(roman: String): String = transliterator.transliterate(roman)

    /** The suggestions from the rules and the word list alone. */
    fun quick(roman: String, nBest: Int = limit): List<String> {
        return engine.suggest(roman, useRules = true).take(nBest).map { it.text }
    }

    /** The words that start with what the prefix model said. */
    fun completion(roman: String, nBest: Int = limit): List<String> {
        if (prefixModel == null) return emptyList()
        val prefix = prefixModel.transliterate(roman)
        if (prefix.isEmpty()) return emptyList()
        return lexicon.complete(prefix, limit = nBest)
    }

    /** The suggestions with the models in the loop. */
    fun suggest(roman: String, nBest: Int = limit): List<String> {
        val scored = transliterator.scoredCandidates(roman, 5)
        val ranked = engine.suggest(
            roman,
            scored.map { it.first },
            scored.map { it.second },
            completion(roman, 8),
        )
        return ranked.take(nBest).map { it.text }
    }
}
