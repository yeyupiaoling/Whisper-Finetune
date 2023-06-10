package com.yeyupiaoling.whisper

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val startRecordBtn = findViewById<Button>(R.id.start_record_activity_btn)
        val startFileBtn = findViewById<Button>(R.id.start_file_activity_btn)
        val testModelBtn = findViewById<Button>(R.id.start_test_activity_btn)

        startRecordBtn.setOnClickListener { view: View? ->
            val intent = Intent(this@MainActivity, RecordActivity::class.java)
            startActivity(intent)
        }
        startFileBtn.setOnClickListener { view: View? ->
            val intent = Intent(this@MainActivity, AudioFileActivity::class.java)
            startActivity(intent)
        }
        testModelBtn.setOnClickListener { view: View? ->
            val intent = Intent(this@MainActivity, TestActivity::class.java)
            startActivity(intent)
        }
    }
}