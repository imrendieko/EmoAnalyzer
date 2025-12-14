Deploy ke Vercel (Ringkas)

Langkah singkat untuk deploy proyek Flask ini ke Vercel:

1. Pastikan `vercel` CLI terpasang dan Anda sudah login:

```bash
npm i -g vercel
vercel login
```

2. Di folder proyek (`d:\TugasSTKI`), jalankan:

```bash
vercel --prod
```

Vercel akan membaca `vercel.json` dan membangun `api/index.py` menggunakan `@vercel/python`.

Catatan dan rekomendasi penyederhanaan:
- `requirements.txt` menyertakan paket yang dibutuhkan saja. Beberapa paket (pandas, scikit-learn, nltk) besar ukuran instalasinya — jika ukuran memutus deploy pada Vercel, pertimbangkan menggunakan endpoint ringan atau externalizing berat (mis. memanggil model dari server lain).
- Static files dan template (`templates/`) disertakan dalam bundle yang dikirim ke fungsi WSGI.

File penting yang ditambahkan:
- `api/index.py` — WSGI entrypoint minimal untuk Vercel.
- `vercel.json` — konfigurasi routing/build Vercel.
- `runtime.txt` — versi python yang digunakan.

Jika Anda ingin saya mencoba deploy otomatis ke Vercel dari mesin ini (membutuhkan akses token/akun), beri tahu — atau saya bisa membantu membuat bundle Docker/Heroku sebagai alternatif.
