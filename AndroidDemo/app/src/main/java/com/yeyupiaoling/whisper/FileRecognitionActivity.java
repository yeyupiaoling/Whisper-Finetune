package com.yeyupiaoling.whisper;

import static com.yeyupiaoling.whisper.UtilsKt.decodeWaveFile;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;

public class FileRecognitionActivity extends AppCompatActivity {
    private static final String TAG = Utils.class.getName();
    private WhisperContext whisperContext;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_file_recognition);

        // 请求权限
        if (!hasPermission()) {
            requestPermission();
        }

        Button selectAudioBtn = findViewById(R.id.select_audio_btn);
        selectAudioBtn.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("audio/*");
            startActivityForResult(intent, 1);
        });
        whisperContext = WhisperContext.createContextFromAsset(getApplicationContext().getAssets(), "models/ggml-model.bin");
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        String audioPath;
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == 1) {
                if (data == null) {
                    Log.w("onActivityResult", "user photo data is null");
                    return;
                }
                Uri image_uri = data.getData();
                audioPath = Utils.getPathFromURI(FileRecognitionActivity.this, image_uri);
                try {
                    File file = new File(audioPath);
                    float[] audioData = decodeWaveFile(file);
                    String text = whisperContext.transcribeData(audioData);
                    Log.d(TAG, text);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    // check had permission
    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            return checkSelfPermission(Manifest.permission.READ_MEDIA_AUDIO) == PackageManager.PERMISSION_GRANTED;
        }else {
            return checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        }
    }

    // request permission
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requestPermissions(new String[]{Manifest.permission.READ_MEDIA_AUDIO}, 1);
        }else {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
    }
}