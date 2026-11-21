#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
// Time GPU work on buffers substantially larger than the system cache.
int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 2;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSUInteger bytes = 128 * 1024 * 1024, n = 4096;
        MTLResourceOptions mode = MTLResourceStorageModeShared;
        id<MTLBuffer> a = [device newBufferWithLength:bytes options:mode];
        id<MTLBuffer> b = [device newBufferWithLength:bytes options:mode];
        id<MTLBuffer> c = [device newBufferWithLength:bytes options:mode];
        if (!a || !b || !c) return 3;
        uint16_t *src = a.contents;
        for (NSUInteger i=0; i<bytes/2; i++) src[i] = 0x2400 + i%1024;
        memcpy(b.contents, a.contents, bytes);
        MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:n rowBytes:2*n dataType:MPSDataTypeFloat16];
        MPSMatrix *ma = [[MPSMatrix alloc] initWithBuffer:a descriptor:desc];
        MPSMatrix *mb = [[MPSMatrix alloc] initWithBuffer:b descriptor:desc];
        MPSMatrix *mc = [[MPSMatrix alloc] initWithBuffer:c descriptor:desc];
        MPSMatrixMultiplication *gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:NO resultRows:n resultColumns:n interiorColumns:n alpha:1 beta:0];
        for (int stage=0;stage<2;stage++) {
            for (int rep=0;rep<9;rep++) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                if (stage==0) {
                    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
                    for(int i=0;i<16;i++) [enc copyFromBuffer:(i%2?b:a) sourceOffset:0 toBuffer:(i%2?a:b) destinationOffset:0 size:bytes];
                    [enc endEncoding];
                } else {
                    for(int i=0;i<4;i++) [gemm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
                }
                [cb commit]; [cb waitUntilCompleted];
                if(cb.status!=MTLCommandBufferStatusCompleted) return 4;
                double t=cb.GPUEndTime-cb.GPUStartTime;
                if(t<=0) return 5;
                printf("{\"metric\":\"%s\",\"warmup\":%s,\"seconds\":%.9f,\"value\":%.5f}\n",stage?"fp16_tflops":"blit_read_write_GBps",rep<2?"true":"false",t,stage?8.0*n*n*n/t/1e12:32.0*bytes/t/1e9);
            }
        }
        if (memcmp(a.contents,b.contents,bytes) || ((__fp16 *)c.contents)[0]<=0) return 6;
    }
}
