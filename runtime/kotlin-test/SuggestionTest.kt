package com.mangokeyboard.transliterator

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
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
        val directory = File("../../data/processed/lexicon")
        assertTrue("the word list is missing, run scripts.build_lexicon", directory.isDirectory)
        return Lexicon().load(directory)
    }

    private fun ngram(): CharNgram {
        val file = File("../../data/processed/ngram.txt")
        assertTrue("the language model is missing, run scripts.train_ngram", file.exists())
        return CharNgram.load(file)
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
        assertTrue(model.score("കേരളം") > model.score("ക്സസ്സ്"))
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
        val ranked = engine.suggest("ente", listOf("ക്സസ്സ്", "ഒരു"), listOf(-1.0, -2.0))
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
