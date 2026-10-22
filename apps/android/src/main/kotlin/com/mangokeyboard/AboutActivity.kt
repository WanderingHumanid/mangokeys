package com.mangokeyboard

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.TextView

/**
 * The about screen of the keyboard.
 *
 * It says what the app is, what is inside it, where the data came from and
 * where to read more. The links open the browser.
 */
class AboutActivity : Activity() {

    private val repository = "https://github.com/WanderingHumanid/mangokeys"
    private val report = "$repository/blob/main/REPORT.md"
    private val release = "$repository/releases/latest"
    private val licence = "$repository/blob/main/LICENSE"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val settings = KeyboardSettings(this)
        setTheme(if (settings.darkMode) R.style.Theme_Manglish_Dark else R.style.Theme_Manglish)
        setContentView(R.layout.about)

        findViewById<View>(R.id.button_back).setOnClickListener { finish() }
        findViewById<View>(R.id.row_repository).setOnClickListener { open(repository) }
        findViewById<View>(R.id.row_report).setOnClickListener { open(report) }
        findViewById<View>(R.id.row_release).setOnClickListener { open(release) }
        findViewById<View>(R.id.row_licence).setOnClickListener { open(licence) }

        findViewById<TextView>(R.id.version).text = "Version ${versionName()}"
    }

    /** The version of the app, as it is in the package. */
    private fun versionName(): String {
        return try {
            val info = packageManager.getPackageInfo(packageName, 0)
            info.versionName ?: ""
        } catch (error: PackageManager.NameNotFoundException) {
            ""
        }
    }

    private fun open(url: String) {
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
        if (intent.resolveActivity(packageManager) != null) startActivity(intent)
    }
}
