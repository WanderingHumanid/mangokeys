package com.mangokeyboard.transliterator

/**
 * The ranker of the keyboard on the phone.
 *
 * The rules and the model both propose words, the word list turns them into
 * real words, and the language model and the corpus frequency decide which
 * one comes first. The same code runs in the browser.
 */
class SuggestionEngine(
    private val lexicon: Lexicon,
    private val ngram: CharNgram?,
    private val weights: Weights = Weights(),
) {
    /** One word the keyboard is considering. */
    data class Candidate(
        val text: String,
        val source: String,
        var modelScore: Double = 0.0,
        var lmScore: Double = 0.0,
        var count: Int = 0,
        var total: Double = 0.0,
    )

    data class Weights(
        val model: Double = 1.0,
        val language: Double = 1.0,
        val frequency: Double = 0.2,
        val unknown: Double = -2.0,
        val rulePenalty: Double = -12.0,
        val limit: Int = 8,
    )

    /**
     * Rank the words that a Manglish string could mean.
     *
     * The model candidates come with the score the model gave them, so that
     * the ranking can prefer the answer the model is most sure about.
     */
    fun suggest(
        roman: String,
        modelCandidates: List<String> = emptyList(),
        modelScores: List<Double> = emptyList(),
        useRules: Boolean = true,
    ): List<Candidate> {
        val text = roman.trim()
        if (text.isEmpty()) return emptyList()

        val pool = LinkedHashMap<String, Candidate>()
        modelCandidates.forEachIndexed { index, word ->
            if (word.isEmpty()) return@forEachIndexed
            val score = if (index < modelScores.size) modelScores[index] else 0.0
            offer(pool, word, "model", score)
        }
        if (useRules) {
            for (word in RuleEngine().convert(text).take(4)) {
                offer(pool, word, "rules", weights.rulePenalty)
            }
        }

        for (candidate in pool.values) {
            candidate.count = lexicon.count(candidate.text)
            candidate.lmScore = ngram?.meanScore(candidate.text) ?: 0.0
            candidate.total = weights.model * candidate.modelScore +
                weights.language * candidate.lmScore +
                weights.frequency * Math.log1p(candidate.count.toDouble())
            if (candidate.count == 0) candidate.total += weights.unknown
        }

        return pool.values.sortedByDescending { it.total }.take(weights.limit)
    }

    private fun offer(
        pool: LinkedHashMap<String, Candidate>,
        text: String,
        source: String,
        score: Double,
    ) {
        val known = pool[text]
        if (known != null) {
            known.modelScore = maxOf(known.modelScore, score)
            return
        }
        pool[text] = Candidate(text, source, modelScore = score)
        for (word in lexicon.expand(text, 4)) {
            if (word == text) continue
            val neighbour = pool[word]
            if (neighbour != null) {
                neighbour.modelScore = maxOf(neighbour.modelScore, score)
            } else {
                pool[word] = Candidate(word, source, modelScore = score)
            }
        }
    }
}
