package com.mangokeyboard.transliterator

/**
 * Unicode normalization for Malayalam.
 *
 * Malayalam has two spellings for the vowel-less chillu forms. The legacy
 * spelling is consonant plus virama plus anusvara, the modern one is a single
 * codepoint. Fonts render both but an input method should only emit the
 * atomic forms, so the legacy ones are rewritten here.
 *
 * The zero width joiner and non joiner carry meaning and are left alone. A
 * chillu is consonant plus virama plus joiner, an explicit virama is
 * consonant plus virama plus non joiner, and those are different words.
 */
object Normalizer {

    private const val VIRAMA = "്"
    private const val ANUSVARA = "ം"
    private const val ZWJ = "‍"
    private const val ZWNJ = "‌"

    private val LEGACY_CHILLU = mapOf(
        "ണ" to "ൺ",
        "ന" to "ൻ",
        "ര" to "ർ",
        "ല" to "ൽ",
        "ള" to "ൾ",
        "ക" to "ൿ",
    )

    private val QUOTES = mapOf(
        "‘" to "'", "’" to "'", "‚" to "'", "‛" to "'",
        "“" to "\"", "”" to "\"", "„" to "\"",
        "«" to "\"", "»" to "\"",
    )

    private val DASHES = mapOf(
        "‐" to "-", "‑" to "-", "‒" to "-",
        "–" to "-", "—" to "-", "―" to "-", "−" to "-",
    )

    private val ZERO_WIDTH = charArrayOf('\u200B', '\u2060', '\uFEFF').toSet()

    fun normalizeMalayalam(text: String): String {
        var out = java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC)
        for ((consonant, chillu) in LEGACY_CHILLU) {
            out = out.replace(consonant + VIRAMA + ANUSVARA, chillu)
            out = out.replace(consonant + VIRAMA + ZWJ, chillu)
        }
        return out
    }

    fun normalizeRoman(text: String): String {
        var out = java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC)
        for ((from, to) in QUOTES) out = out.replace(from, to)
        for ((from, to) in DASHES) out = out.replace(from, to)
        out = out.filter { it !in ZERO_WIDTH }
        return out.replace(Regex("\\s+"), " ").trim()
    }
}
