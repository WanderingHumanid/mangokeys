package com.mangokeyboard.transliterator

/**
 * The rule based Manglish to Malayalam converter.
 *
 * This is the instant path of the keyboard. It cannot resolve the ambiguity
 * of Malayalam orthography, so it offers several readings of a word and lets
 * the model and the word list choose between them.
 *
 * It is the same engine as the python one in model/src/rules.py, so that the
 * three runtimes agree.
 */
class RuleEngine {

    /** The candidate Malayalam spellings, the best guess first. */
    fun convert(text: String, maxCandidates: Int = 12): List<String> {
        val clean = text.trim()
        if (clean.isEmpty()) return emptyList()
        val segments = parse(clean)
        val out = LinkedHashSet<String>()
        out.add(render(segments))
        for (index in segments.indices) {
            val segment = segments[index]
            if (segment.kind != "consonant") continue
            for (alternative in ALTERNATIVES[segment.key].orEmpty()) {
                if (alternative == segment.base) continue
                val copy = segments.toMutableList()
                copy[index] = segment.copy(base = alternative, text = alternative)
                out.add(render(copy))
            }
        }
        return out.toList().take(maxCandidates)
    }

    /** The best guess for a word. */
    fun convertTop(text: String): String = convert(text, 1).firstOrNull() ?: ""

    private data class Segment(
        val kind: String,
        val text: String,
        val key: String = "",
        val base: String = "",
        val doubled: Boolean = false,
        val signKey: String? = null,
        val sign: String? = null,
    )

    private fun parse(text: String): List<Segment> {
        val segments = ArrayList<Segment>()
        var index = 0
        while (index < text.length) {
            val cluster = match(text, index, CLUSTER_KEYS)
            if (cluster != null) {
                val sign = matchSign(text, index + cluster.length)
                segments.add(
                    Segment(
                        kind = "consonant",
                        text = CLUSTERS[cluster]!!,
                        key = cluster,
                        base = CLUSTERS[cluster]!!,
                        signKey = sign.second,
                        sign = sign.first,
                    ),
                )
                index += cluster.length + (sign.second?.length ?: 0)
                continue
            }

            val consonant = match(text, index, CONSONANT_KEYS)
            if (consonant != null) {
                val doubled = consonant.length == 1 && text.getOrNull(index + 1) == consonant[0]
                var base = CONSONANTS[consonant]!!
                val advance = if (doubled) 2 else consonant.length
                if (doubled) base = base + VIRAMA + base
                val sign = matchSign(text, index + advance)
                segments.add(
                    Segment(
                        kind = "consonant",
                        text = base,
                        key = consonant,
                        base = base,
                        doubled = doubled,
                        signKey = sign.second,
                        sign = sign.first,
                    ),
                )
                index += advance + (sign.second?.length ?: 0)
                continue
            }

            val vowel = match(text, index, VOWEL_KEYS)
            if (vowel != null) {
                segments.add(
                    Segment(kind = "vowel", text = INDEPENDENT_VOWELS[vowel]!!, key = vowel),
                )
                index += vowel.length
                continue
            }

            segments.add(Segment(kind = "other", text = text[index].toString()))
            index += 1
        }
        return segments
    }

    private fun match(text: String, index: Int, keys: List<String>): String? {
        for (key in keys) {
            if (text.startsWith(key, index)) return key
        }
        return null
    }

    private fun matchSign(text: String, index: Int): Pair<String?, String?> {
        val key = match(text, index, SIGN_KEYS) ?: return null to null
        return VOWEL_SIGNS[key] to key
    }

    private fun render(segments: List<Segment>): String {
        val out = StringBuilder()
        val last = segments.size - 1
        for (index in segments.indices) {
            val segment = segments[index]
            if (segment.kind == "vowel" || segment.kind == "other") {
                out.append(segment.text)
                continue
            }
            if (segment.signKey != null) {
                out.append(segment.base).append(segment.sign ?: "")
                continue
            }
            if (index < last) {
                out.append(segment.base)
                if (segments[index + 1].kind == "consonant") out.append(VIRAMA)
                continue
            }
            if (!segment.doubled && segment.base == "മ") {
                out.append(ANUSVARA)
                continue
            }
            if (!segment.doubled && CHILLUS.containsKey(segment.base)) {
                out.append(CHILLUS[segment.base])
                continue
            }
            out.append(segment.base).append(VIRAMA)
        }
        return out.toString()
    }

    companion object {
        private const val VIRAMA = "്"
        private const val ANUSVARA = "ം"

        private val INDEPENDENT_VOWELS = mapOf(
            "a" to "അ", "aa" to "ആ", "i" to "ഇ", "ii" to "ഈ",
            "u" to "ഉ", "uu" to "ഊ", "e" to "എ", "E" to "ഏ",
            "ai" to "ഐ", "o" to "ഒ", "O" to "ഓ", "au" to "ഔ",
        )

        private val VOWEL_SIGNS = mapOf(
            "a" to "", "aa" to "ാ", "i" to "ി", "ii" to "ീ",
            "u" to "ു", "uu" to "ൂ", "e" to "െ", "ee" to "േ",
            "E" to "േ", "ai" to "ൈ", "o" to "ൊ", "oo" to "ോ",
            "O" to "ോ", "au" to "ൗ",
        )

        private val CONSONANTS = mapOf(
            "k" to "ക", "kh" to "ഖ", "g" to "ഗ", "gh" to "ഘ", "ng" to "ങ",
            "ch" to "ച", "chh" to "ഛ", "j" to "ജ", "jh" to "ഝ", "nj" to "ഞ",
            "T" to "ട", "Th" to "ഠ", "D" to "ഡ", "Dh" to "ഢ", "N" to "ണ",
            "t" to "ത", "th" to "ത", "thh" to "ഥ", "d" to "ദ", "dh" to "ധ",
            "n" to "ന", "p" to "പ", "ph" to "ഫ", "b" to "ബ", "bh" to "ഭ",
            "m" to "മ", "y" to "യ", "r" to "ര", "l" to "ല", "v" to "വ",
            "w" to "വ", "sh" to "ശ", "Sh" to "ഷ", "s" to "സ", "h" to "ഹ",
            "L" to "ള", "zh" to "ഴ", "R" to "റ", "rr" to "റ",
        )

        private val CLUSTERS = mapOf(
            "nt" to "ന്റ", "nk" to "ങ്ക", "nch" to "ഞ്ച", "ksh" to "ക്ഷ",
            "nd" to "ന്ദ", "mb" to "മ്പ", "nth" to "ന്ത",
        )

        private val CHILLUS = mapOf(
            "ണ" to "ൺ", "ന" to "ൻ", "ര" to "ർ",
            "ല" to "ൽ", "ള" to "ൾ", "ക" to "ൿ",
        )

        // The readings a person could have meant instead of the primary one.
        private val ALTERNATIVES = mapOf(
            "t" to listOf("ട", "റ", "റ്റ"),
            "th" to listOf("ഥ", "ഠ"),
            "d" to listOf("ഡ", "ധ"),
            "dh" to listOf("ഢ", "ഥ"),
            "n" to listOf("ണ"),
            "l" to listOf("ള"),
            "r" to listOf("റ"),
            "s" to listOf("ശ", "ഷ"),
            "sh" to listOf("ഷ", "സ"),
            "ng" to listOf("ങ്ങ"),
            "kk" to listOf("ക്ക"),
            "nd" to listOf("ണ്ട"),
            "nth" to listOf("ന്ത", "ന്റ"),
            "T" to listOf("ത"),
            "Th" to listOf("ഥ", "ത"),
            "D" to listOf("ദ"),
            "Dh" to listOf("ധ", "ദ"),
            "N" to listOf("ന"),
            "L" to listOf("ല"),
            "R" to listOf("ര"),
            "Sh" to listOf("സ"),
        )

        private val CLUSTER_KEYS = CLUSTERS.keys.sortedByDescending { it.length }
        private val CONSONANT_KEYS = CONSONANTS.keys.sortedByDescending { it.length }
        private val VOWEL_KEYS = INDEPENDENT_VOWELS.keys.sortedByDescending { it.length }
        private val SIGN_KEYS = VOWEL_SIGNS.keys.sortedByDescending { it.length }
    }
}
