# CPU image: `docker build -t pocket-tts .`
# then `docker run -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface pocket-tts`
FROM rust:1.97-bookworm AS builder
WORKDIR /build
COPY . .
RUN cargo build --release --locked -p pocket-tts-cli

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/pocket-tts /usr/local/bin/pocket-tts
ENV POCKET_TTS_HOST=0.0.0.0 \
    POCKET_TTS_PORT=8000 \
    POCKET_TTS_VARIANT=english
EXPOSE 8000
ENTRYPOINT ["pocket-tts"]
CMD ["serve"]
