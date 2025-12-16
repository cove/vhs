#
# Generates half-resolution proxy MP4 files from archival MKV sources, embedding chapter metadata
# for easier preview and faster access while preserving original archives.
#
from common import *

def main():
    print(f"Generating ½-size PROXY → {ARCHIVE_DIR}\n")
    count = 0
    for src in ARCHIVE_DIR.glob("*.mkv"):
        archive = src.stem
        proxy = ARCHIVE_DIR / f"{archive}_proxy.mp4"
        ffmetadata_path = METADATA_DIR / archive / "chapters.ffmetadata"

        if proxy.exists():
            print(f"Skipping {proxy} (already processed)")
            continue

        print(f"Processing: {src.name} → {proxy.name}")
        run([FFMPEG_BIN,
             "-nostdin",
             "-v", "error",
             "-i", str(src),
             "-i", str(ffmetadata_path),
             "-map", "0:v:0",
             "-map", "0:a:0",
             "-map_metadata", "1",
             "-c:v", "libx264", "-preset", "superfast",  "-tune", "fastdecode", "-crf", "28",
             "-x264-params", "keyint=30:min-keyint=1:scenecut=40",
             "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "48k", "-ar", "48000", "-ac", "1",
             "-movflags", "+faststart+use_metadata_tags",
             "-y", str(proxy)])
        count += 1

    print(f"Created {count} proxies.")
    print("All done")

if __name__ == "__main__":
    main()
