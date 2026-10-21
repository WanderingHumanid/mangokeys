package com.mangokeyboard

import android.os.Bundle
import android.widget.TextView

/** A short description of what the keyboard is and what it does not do. */
class AboutActivity : android.app.Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val text = TextView(this)
        text.text = getString(R.string.about_text)
        text.setPadding(48, 48, 48, 48)
        text.textSize = 16f
        setContentView(text)
    }
}
