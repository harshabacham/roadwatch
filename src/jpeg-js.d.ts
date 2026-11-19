declare module 'jpeg-js' {
  export interface RawImageData {
    width: number;
    height: number;
    data: Uint8Array;
  }
  export function decode(
    jpegData: ArrayBuffer | Uint8Array | Buffer,
    opts?: { useTArray?: boolean; formatAsRGBA?: boolean }
  ): RawImageData;
}
