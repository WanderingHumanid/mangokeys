package com.mangokeyboard.transliterator

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * The word list, the language model and the ranker.
 *
 * They have to agree with the python runtime, because the python runtime is
 * what the numbers in the report were produced with.
 */
class SuggestionTest {

    private fun lexicon(): Lexicon {
        val directory = bundle("lexicon")
        assertTrue("the word list is missing, run scripts.build_release_assets", directory.isDirectory)
        return Lexicon().load(directory)
    }

    /** The language model is not in the bundle, so the test is skipped. */
    private fun ngram(): CharNgram? {
        val file = File("../../data/processed/ngram.txt")
        if (!file.exists()) return null
        return CharNgram.load(file)
    }

    /** The committed bundle, or the build folder of a local run. */
    private fun bundle(name: String): File {
        val committed = File("../../assets/release/$name")
        if (committed.exists()) return committed
        return when (name) {
            "lexicon" -> File("../../data/processed/lexicon")
            else -> File("../../runs/demo/$name")
        }
    }

    @Test
    fun findsAWordThatIsInTheList() {
        val words = lexicon()
        assertEquals("ഒരു", words.nearest("ഒരു", limit = 1).first())
    }

    @Test
    fun findsTheWordWithAWrongVowel() {
        val words = lexicon()
        val near = words.nearest("ഒരൂ", limit = 1)
        assertTrue(near.isNotEmpty())
        assertTrue(words.contains(near.first()))
    }

    @Test
    fun completesAPrefix() {
        val found = lexicon().complete("ഒരു")
        assertTrue(found.contains("ഒരു"))
    }

    @Test
    fun reportsTheCorpusCount() {
        assertTrue(lexicon().count("ഒരു") > 0)
    }

    @Test
    fun theLanguageModelPrefersAWordItHasSeen() {
        val model = ngram()
        assumeTrue("the language model is not built", model != null)
        assertTrue(model!!.score("കേരളം") > model.score("ക്സ്സ്സ്"))
    }

    @Test
    fun theRankerPutsTheModelAnswerFirst() {
        val engine = SuggestionEngine(lexicon(), ngram())
        val ranked = engine.suggest("ente", listOf("എൻ്റെ"), listOf(-1.0))
        assertEquals("എൻ്റെ", ranked.first().text)
    }

    @Test
    fun theRankerPrefersARealWord() {
        val engine = SuggestionEngine(lexicon(), ngram())
        val ranked = engine.suggest("ente", listOf("ക്സ്സ്സ്", "ഒരു"), listOf(-1.0, -2.0))
        assertEquals("ഒരു", ranked.first().text)
    }

    @Test
    fun theRuleEngineAgreesWithThePythonOne() {
        val engine = RuleEngine()
        assertEquals("എന്റെ", engine.convertTop("ente"))
        assertEquals("കെരല", engine.convertTop("kerala"))
        assertEquals("കാലം", engine.convertTop("kaalam"))
    }
}
