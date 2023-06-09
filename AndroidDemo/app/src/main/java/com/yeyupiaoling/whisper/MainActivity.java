package com.yeyupiaoling.whisper;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button startRecord = findViewById(R.id.start_record_activity_btn);
        Button startFile = findViewById(R.id.start_file_activity_btn);
        startRecord.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, RecordActivity.class);
            startActivity(intent);
        });
        startFile.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, FileRecognitionActivity.class);
            startActivity(intent);
        });
    }
}