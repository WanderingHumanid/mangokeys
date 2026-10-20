package com.mangokeyboard.transliterator

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The runtime has to agree with the python one, because the python one is what
 * the numbers in the report were produced with.
 */
class TransliteratorTest {

    private fun runtime(): Transliterator {
        val models = File("../../runs/demo/onnx-int8")
        val tokenizer = File("../../runs/ml_mini/tokenizer.json")
        assertTrue("the models are missing, run scripts/build_demo first", models.isDirectory)
        return Transliterator(models, tokenizer.readText(), maxSrc = 48, maxTgt = 64, beam = 4)
    }

    @Test
    fun producesTheExpectedWords() {
        val engine = runtime()
        val expected = mapOf(
            "ente" to "എൻ്റെ",
            "peru" to "പേരു",
            "kerala" to "കേരള",
            "sundari" to "സുന്ദരി",
            "kaalam" to "കാലം",
        )
        for ((roman, malayalam) in expected) {
            assertEquals(roman, malayalam, engine.transliterate(roman))
        }
    }

    @Test
    fun splitsAPhraseIntoWords() {
        val engine = runtime()
        assertEquals("എൻ്റെ പേരു", engine.transliterate("ente peru"))
    }

    @Test
    fun normalizesTheLegacyChilluSpelling() {
        val legacy = "അ" + "വ" + "ന" + "്" + "ം"
        assertEquals("അവൻ", Normalizer.normalizeMalayalam(legacy))
    }

    @Test
    fun keepsTheRomanCase() {
        assertEquals("Ta", Normalizer.normalizeRoman("Ta"))
    }
}
