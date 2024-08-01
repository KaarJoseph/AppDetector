package ups.com.emotionrecognitionapp;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import ups.com.emotionrecognitionapp.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity"; // Definición de TAG
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private ActivityMainBinding binding;
    private Bitmap bitmapI;
    private Bitmap bitmapProcessed;
    private ArrayList<Float> hogDescriptors; // Para almacenar los descriptores HOG

    static {
        System.loadLibrary("emotionrecognitionapp");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        copyAssets();

        Button btnCapture = binding.btnCapture;
        Button btnDetect = binding.btnDetect;
        Button btnHog = binding.btnHog;
        Button btnSend = binding.btnSend;

        btnCapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                abrirCamara();
            }
        });

        btnDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (bitmapI != null) {
                    new DetectFaceTask().execute(bitmapI);
                } else {
                    Toast.makeText(MainActivity.this, "Por favor, capture una imagen primero.", Toast.LENGTH_SHORT).show();
                }
            }
        });

        btnHog.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (bitmapI != null) {
                    new CalculateHogTask().execute(bitmapI);
                } else {
                    Toast.makeText(MainActivity.this, "Por favor, capture una imagen primero.", Toast.LENGTH_SHORT).show();
                }
            }
        });

        btnSend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (bitmapI != null) {
                    enviarImagen(bitmapI);
                }
            }
        });
    }

    private void abrirCamara() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_IMAGE_CAPTURE);
        } else {
            dispatchTakePictureIntent();
        }
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_IMAGE_CAPTURE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                abrirCamara();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");
            if (imageBitmap != null) {
                bitmapI = imageBitmap;
                binding.ivCapturedImage.setImageBitmap(bitmapI);
            }
        }
    }

    private class DetectFaceTask extends AsyncTask<Bitmap, Void, String> {
        @Override
        protected String doInBackground(Bitmap... bitmaps) {
            Bitmap originalBitmap = bitmaps[0];
            bitmapProcessed = originalBitmap.copy(originalBitmap.getConfig(), true); // Crear una copia para el procesamiento
            return detectFeatures(originalBitmap, bitmapProcessed);
        }

        @Override
        protected void onPostExecute(String result) {
            if (result == null || result.isEmpty()) {
                Toast.makeText(MainActivity.this, "Error al detectar características", Toast.LENGTH_SHORT).show();
                return;
            }

            updateUIWithResult(result);
            binding.ivDetectedImage.setImageBitmap(bitmapProcessed); // Mostrar la imagen procesada
        }
    }

    private class CalculateHogTask extends AsyncTask<Bitmap, Void, Boolean> {
        @Override
        protected Boolean doInBackground(Bitmap... bitmaps) {
            Bitmap originalBitmap = bitmaps[0];
            String result = calculateHOG(originalBitmap);

            // Convertir el string resultante en una lista de flotantes
            hogDescriptors = new ArrayList<>();
            String[] descriptorsArray = result.split(" ");
            for (String descriptor : descriptorsArray) {
                hogDescriptors.add(Float.parseFloat(descriptor));
            }
            return result != null && !result.isEmpty();
        }

        @Override
        protected void onPostExecute(Boolean success) {
            if (success) {
                binding.tvHog.setText("Descriptores HOG calculados");
            } else {
                binding.tvHog.setText("No calculados");
            }
        }
    }

    private native String detectFeatures(Bitmap originalBitmap, Bitmap processedBitmap);

    private native String calculateHOG(Bitmap bitmap);

    public void updateUIWithResult(final String result) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                binding.tvResult.setText("Resultado: " + result);
            }
        });
    }

    private void copyAssets() {
        String[] assetFiles = {"haarcascade_frontalface_alt.xml", "haarcascade_eye.xml", "haarcascade_mcs_nose.xml", "haarcascade_mcs_mouth.xml"};
        for (String fileName : assetFiles) {
            try {
                InputStream is = getAssets().open(fileName);
                File outFile = new File(getFilesDir(), fileName);
                FileOutputStream fos = new FileOutputStream(outFile);
                byte[] buffer = new byte[1024];
                int length;
                while ((length = is.read(buffer)) > 0) {
                    fos.write(buffer, 0, length);
                }
                fos.close();
                is.close();
                Log.d(TAG, "Archivo copiado: " + outFile.getAbsolutePath());
            } catch (IOException e) {
                Log.e(TAG, "Failed to copy asset file: " + fileName, e);
            }
        }
    }

    private void enviarImagen(Bitmap bitmap) {
        // Convertir bitmap a base64
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        byte[] byteArray = byteArrayOutputStream.toByteArray();
        String encodedImage = Base64.encodeToString(byteArray, Base64.NO_WRAP); // Funcional hasta aquí

        // Crear JSON
        String featuresJson = detectFeatures(bitmapI, bitmapProcessed);
        String hogDescriptors = calculateHOG(bitmapI);

        // Convertir descriptores HOG a lista de floats
        String[] hogArray = hogDescriptors.split(" ");
        float[] hogFloatArray = new float[hogArray.length];
        for (int i = 0; i < hogArray.length; i++) {
            hogFloatArray[i] = Float.parseFloat(hogArray[i]);
        }

        // Crear el JSON request
        JSONObject jsonRequest = new JSONObject();
        try {
            jsonRequest.put("image", encodedImage);
            jsonRequest.put("features", new JSONObject(featuresJson));
            jsonRequest.put("hog", new JSONArray(hogFloatArray)); // Enviar como JSONArray
        } catch (JSONException e) {
            e.printStackTrace();
        }

        // Enviar solicitud POST
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.create(jsonRequest.toString(), JSON);
        Request request = new Request.Builder()
                .url("http://192.168.1.7:5000/upload")
                .post(body)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                Log.e(TAG, "Error al enviar la imagen: " + e.getMessage());
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    Log.d(TAG, "Imagen enviada con éxito");
                } else {
                    Log.e(TAG, "Error en la respuesta: " + response.message());
                }
            }
        });
    }
}
