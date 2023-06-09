package com.yeyupiaoling.whisper;

import static com.yeyupiaoling.whisper.Utils.writeWAVHeader;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class RecordActivity extends AppCompatActivity {
    // 采样率
    public static final int SAMPLE_RATE = 16000;
    // 声道数
    public static final int CHANNEL = AudioFormat.CHANNEL_IN_MONO;
    // 返回的音频数据的格式
    public static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private AudioRecord audioRecord;
    private boolean mIsRecording = false;
    private int minBufferSize;
    private String wavPath = null;
    private AudioView audioView;

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_record);
        // 请求权限
        if (!hasPermission()) {
            requestPermission();
        }
        wavPath = getExternalFilesDir(Environment.DIRECTORY_MUSIC).getAbsolutePath() + "/test.wav";
        minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, AUDIO_FORMAT);
        audioView = findViewById(R.id.audioView);
        audioView.setStyle(AudioView.ShowStyle.STYLE_HOLLOW_LUMP, AudioView.ShowStyle.STYLE_NOTHING);
        Button mRecordButton = findViewById(R.id.record_button);
        mRecordButton.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_UP) {
                mIsRecording = false;
                stopRecording();
                mRecordButton.setText("按下录音");
            } else if (event.getAction() == MotionEvent.ACTION_DOWN) {
                mIsRecording = true;
                startRecording();
                mRecordButton.setText("录音中...");
            }
            return true;
        });
    }


    private void startRecording() {
        // 准备录音器
        try {
            // 创建录音器
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                requestPermission();
                return;
            }
            audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL, AUDIO_FORMAT, minBufferSize);
        } catch (IllegalStateException e) {
            e.printStackTrace();
        }
        // 开启一个线程将录音数据写入文件
        Thread recordingAudioThread = new Thread(() -> {
            try {
                writeAudioDataToWavFile(wavPath);
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        recordingAudioThread.start();
        // 启动录音器
        audioRecord.startRecording();
        audioView.setVisibility(View.VISIBLE);
    }

    private void stopRecording() {
        // 停止录音器
        audioRecord.stop();
        audioRecord.release();
        audioRecord = null;
        audioView.setVisibility(View.GONE);
        // 开始识别
        try {
            File file = new File(wavPath);
            float[] audioData = decodeWaveFile(file);
            String text = whisperContext.transcribeData(audioData);
            Log.d(TAG, text);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 保存音频
    private void writeAudioDataToWavFile(String wavPath) throws IOException {
        File file = new File(wavPath);
        if (file.exists()) {
            file.delete();
        }
        FileOutputStream fos = new FileOutputStream(file);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buffer = new byte[minBufferSize];
        audioRecord.startRecording();
        while (mIsRecording) {
            int readSize = audioRecord.read(buffer, 0, minBufferSize);
            if (readSize > 0) {
                bos.write(buffer, 0, readSize);
                audioView.post(() -> audioView.setWaveData(buffer));
            }
        }
        byte[] audioData = bos.toByteArray();
        long totalAudioLen = audioData.length;
        long totalDataLen = totalAudioLen + 36;
        writeWAVHeader(fos, totalAudioLen, totalDataLen, SAMPLE_RATE, 1, 16);
        fos.write(audioData);
    }

    // check had permission
    private boolean hasPermission() {
        return checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED &&
                checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    // request permission
    private void requestPermission() {
        requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO,
                Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
    }
}