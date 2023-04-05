Buffer<float4> g_InputBuffer : register(t0);
RWTexture2D<float4> g_OutputTexture : register(u0);
cbuffer cb0 : register(b0)
{
    uint4 g_Parameters;
};

[numthreads(8, 8, 1)]
void main(uint3 groupCoord: SV_GroupThreadID, uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 textureSize;
    g_OutputTexture.GetDimensions(textureSize.x, textureSize.y);

    uint2 pixelCoord = dispatchThreadId.xy;
    if (pixelCoord.x < textureSize.x && pixelCoord.y < textureSize.y)
    {
        float4 pixelColor = g_InputBuffer[pixelCoord.y * textureSize.x + pixelCoord.x];
        g_OutputTexture[int2(pixelCoord)] = pixelColor;
    }
}