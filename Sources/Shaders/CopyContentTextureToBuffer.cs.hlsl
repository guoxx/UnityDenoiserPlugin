RWBuffer<float4> g_OutputBuffer : register(u0);
Texture2D<float4> g_InputTexture : register(t0);
cbuffer cb0 : register(b0)
{
    uint4 g_Parameters;
};

[numthreads(8, 8, 1)]
void main(uint3 groupCoord: SV_GroupThreadID, uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 textureSize;
    g_InputTexture.GetDimensions(textureSize.x, textureSize.y);

    uint2 pixelCoord = dispatchThreadId.xy;
    if (pixelCoord.x < textureSize.x && pixelCoord.y < textureSize.y)
    {
        float4 pixelColor = g_InputTexture.Load(int3(pixelCoord, 0));
        g_OutputBuffer[pixelCoord.y * textureSize.x + pixelCoord.x] = pixelColor;
    }
}