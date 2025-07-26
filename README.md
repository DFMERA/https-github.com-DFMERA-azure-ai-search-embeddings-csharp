# Azure AI Search: Cómo optimizar la búsqueda para tus aplicaciones RAG

Este repositorio tambien está disponible en Python: [azure-ai-search-embeddings](https://github.com/DFMERA/azure-ai-search-embeddings)

## ¿Cómo funciona Azure AI Search?
[Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) es un servicio de búsqueda de información que utiliza IA para mejorar la relevancia y precisión de los resultados.

Azure AI Search permite indexar información utilizando vectores, lo que significa que puede buscar no solo por palabras clave, sino también por conceptos y relaciones entre términos. Esto es especialmente útil para aplicaciones de [Retrival-Augmented Generation (RAG)](https://azure.microsoft.com/en-us/products/ai-services/ai-search).

## Ejemplo de un proceso de indexación

![Proceso de indexación](https://github.com/Azure-Samples/azure-search-openai-demo/raw/main/docs/images/diagram_prepdocs.png)

## ¿Qué es una indexación de vectores (vector embeddings)?

Un embedding vectorial es una representación numérica de un objeto, como un texto o una imagen, que captura sus características semánticas. En el contexto de Azure AI Search, los embeddings permiten que el motor de búsqueda comprenda mejor el significado y la relación entre diferentes términos.

**Ejemplo:** 

* "Perro" => [0.1, 0.2, 0.3, ...]
* "Animal" => [0.4, 0.5, 0.6, ...]

En este ejemplo, los vectores de "Perro" y "Animal" están cerca en el espacio vectorial, lo que indica que están relacionados semánticamente. Esto permite que Azure AI Search encuentre resultados relevantes incluso si las palabras exactas no coinciden.

## Generación de vector embeddings con OpenAI

### Modelos para generación de embeddings

- **text-embedding-ada-002**: Un modelo optimizado para tareas de embeddings, que combina un buen equilibrio entre velocidad y precisión.
- **text-embedding-3-small**: Un modelo de tamaño pequeño, ideal para aplicaciones que requieren una generación rápida de embeddings con un costo reducido.
- **text-embedding-3-large**: Un modelo de tamaño grande, que ofrece una mayor precisión en la generación de embeddings, pero a un costo más alto y con un tiempo de respuesta más lento.

### Ejemplo de generación de embeddings

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenAI;
using OpenAI.Embeddings;
using DotNetEnv;
using System.ClientModel;

// Load environment variables
Env.Load();

const string MODEL_NAME = "openai/text-embedding-3-small";

var endpoint = "https://models.github.ai/inference";
var credential = System.Environment.GetEnvironmentVariable("GITHUB_TOKEN");

var openAIOptions = new OpenAIClientOptions()
{
    Endpoint = new Uri(endpoint)
};

EmbeddingClient client = new(MODEL_NAME, new ApiKeyCredential(credential), openAIOptions);

string contentInput = "Hoja de vida: Lionel Messi. Futbolista argentino, considerado uno de los mejores jugadores de fútbol de todos los tiempos.";

public static ReadOnlyMemory<float> PrintAndReturnEmbeddings(OpenAIEmbeddingCollection response)
{
    var embedding = response.FirstOrDefault();
    ReadOnlyMemory<float> vector = embedding.ToFloats();
    int length = vector.Length;
    System.Console.Write($"data[{embedding.Index}]: length={length}, ");
    System.Console.Write($"[{vector.Span[0]}, {vector.Span[1]}, ..., ");
    System.Console.WriteLine($"{vector.Span[length - 2]}, {vector.Span[length - 1]}]");
    return vector;
}

OpenAIEmbeddingCollection response = client.GenerateEmbeddings(
    new List<string>{
        contentInput
    }
);

var embeddingsResponse1 = PrintAndReturnEmbeddings(response);
```
```
1536
[0.01875680685043335, -0.01207759603857994, -0.014209441840648651,...]
```

### Comparación de similitud entre vectores

Para comparar la similitud entre dos vectores, se puede utilizar una métrica de distancia como la "distancia coseno". Esta métrica mide el ángulo entre dos vectores y es útil para determinar cuán similares son en términos de dirección.

**Por ejemplo:** Supongamos que queremos comparar los vectores del texto generado conteriormente con una frase de consulta como "Diego Zumárraga Mera". Podemos calcular la similitud entre los dos vectores generados por OpenAI.

Primero generamos los embeddings para ambas frases

```c#
var response2 = client.GenerateEmbeddings(
    new List<string>{
        "Diego Zumárraga Mera"
    }
);

var embeddingsResponse2 = PrintAndReturnEmbeddings(response2);
```

Luego, calculamos la distancia coseno entre los dos vectores:

```c#
public static double CosineSimilarity(float[] v1, float[] v2)
{
    if (v1.Length != v2.Length)
        throw new ArgumentException("Vectors must have the same length");
    
    double dotProduct = v1.Zip(v2, (a, b) => (double)a * b).Sum();
    
    double magnitude1 = Math.Sqrt(v1.Sum(a => (double)a * a));
    double magnitude2 = Math.Sqrt(v2.Sum(a => (double)a * a));
    double magnitude = magnitude1 * magnitude2;
    
    return dotProduct / magnitude;
}

// Compare the two vectors
double similarity = CosineSimilarity(embeddingsResponse1.ToArray(), embeddingsResponse2.ToArray());
Console.WriteLine($"Similarity: {similarity:F4}");
```
```
Similarity: 0.1799
```

Como podemos ver, la similitud entre los dos vectores es de aproximadamente 0.1799, lo que indica que no son muy similares semánticamente.

Ahora cambiemos la frase de consulta a "Hoja de Vida: Diego Zumárraga Mera" y volvamos a calcular la similitud:

```c#
var response3 = client.GenerateEmbeddings(
    new List<string>{
        "Resume la hoja de vida de Diego Zumárraga Mera"
    }
);

var embeddingsResponse3 = PrintAndReturnEmbeddings(response3);

// Compare the two vectors
double similarity2 = CosineSimilarity(embeddingsResponse1.ToArray(), embeddingsResponse3.ToArray());
Console.WriteLine($"Similarity: {similarity2:F4}");
```
```
Similarity: 0.3429
```
Código completo en el notebook: [VectorEmbeddingsNet.ipynb](VectorEmbeddings.ipynb)

La similitud entre los dos vectores aumentó a casi el doble, lo que indica que ahora son más similares semánticamente. Esto es porque la frase "Hoja de Vida" genera una similitud con cualquier texto que contenga esa frase.

### Conlusión
Si queremos desarrollar una solución RAG para indexar y buscar documentos como hojas de vida, esta puede tener problemas para buscar una hoja de vida específica si la consulta incluye frases comunes como "Hoja de Vida". Esto se debe a que Azure AI Search puede devolver resultados que contienen esa frase, pero no necesariamente son relevantes para la consulta.

## Solución: Configurar la búsqueda semántica en Azure AI Search

Para mejorar la relevancia de los resultados, podemos configurar la búsqueda semántica en Azure AI Search para que tome en cuenta otras características del índice como el título del documento o palabras clave adicionales.

### Configuración de la búsqueda semántica

Al momento de crear un índice en Azure AI Search podemos crear dos campos para utilizarlos como `title` y `keywords`. Estos campos pueden ser utilizados luego en la configuración de `semanticSearch` para mejorar la relevancia de los resultados.

```c#
// Definir el índice con campo vectorial y configuración semántica
SearchIndex CreateIndexDefinition(string name)
{
    var fields = new List<SearchField>
    {
        new SimpleField("id", SearchFieldDataType.String) { IsKey = true },
        new SearchableField("title") {
                    IsFacetable = true,
                    IsFilterable = true,
                    AnalyzerName = LexicalAnalyzerName.EsMicrosoft,
                },
        new SearchableField("content") { AnalyzerName = LexicalAnalyzerName.EsMicrosoft },
        new SearchableField("keywords") {
                    IsFacetable = true,
                    IsFilterable = true,
                    AnalyzerName = LexicalAnalyzerName.EsMicrosoft,
                },
        new SearchField("embedding", SearchFieldDataType.Collection(SearchFieldDataType.Single))
        {
            IsSearchable = true,
            VectorSearchDimensions = 1536,
            VectorSearchProfileName = "myHnswProfile"
        }
    };

    var semanticConfig = new SemanticConfiguration("default",
        new() { ContentFields = { new SemanticField("content") },
                TitleField = new SemanticField("title"),
                KeywordsFields = { new SemanticField("keywords") },
            }
    );

    var vectorSearch = new VectorSearch()
    {
        Algorithms =  {
            new HnswAlgorithmConfiguration("myHnsw")
        },
        Profiles = {
            new VectorSearchProfile("myHnswProfile", "myHnsw")
        }
    };
    

    var semanticSearch = new SemanticSearch() { Configurations = {semanticConfig} };

    return new SearchIndex(name)
    {
        Fields = fields,
        SemanticSearch = semanticSearch,
        VectorSearch = vectorSearch
    };
}
```

Cuadno indexemos un documento, debemos asegurarnos de incluir los campos `title` y `keywords`. Es estos campos podemos evitar incluir frases comunes y repetitivas como: "Hoja de Vida", "Manual de Usuario", etc. En su lugar, podemos incluir palabras clave que sean más específicas para el contenido del documento.

```c#
const string MODEL_NAME = "openai/text-embedding-3-small";
var openAIOptions = new OpenAIClientOptions()
{
    Endpoint = new Uri("https://models.github.ai/inference")
};
EmbeddingClient embeddingClient = new(MODEL_NAME, new ApiKeyCredential(githubToken), openAIOptions);

var contentInput = new[]
{
    new { Title = "Lionel Messi", Content = "Hoja de vida: Lionel Messi. Futbolista argentino, considerado uno de los mejores jugadores de fútbol de todos los tiempos." },
    new { Title = "Diego Zumárraga Mera", Content = "Hoja de Vida: Diego Zumárraga Mera. Ingeniero de software con experiencia en desarrollo de aplicaciones web y móviles, apasionado por la inteligencia artificial y el aprendizaje automático." }
};

var docs = new List<Dictionary<string, object>>();
for (int ix = 0; ix < contentInput.Length; ix++)
{
    var item = contentInput[ix];
    Console.WriteLine($"Procesando documento {ix}: {item.Content}");
    OpenAIEmbeddingCollection response = embeddingClient.GenerateEmbeddings(new List<string> { item.Content });
    var embedding = response.FirstOrDefault().ToFloats().ToArray();
    Console.WriteLine($"Embedding length: {embedding.Length}");

    docs.Add(new Dictionary<string, object>
    {
        ["id"] = $"doc-{ix}",
        ["title"] = item.Title,
        ["content"] = item.Content,
        ["keywords"] = item.Title.Replace(" ", ", "),
        ["embedding"] = embedding
    });
}
Console.WriteLine($"Generados {docs.Count} documentos para indexar.");
```
Código completo en el notebook: [AzureAISearchIndexNet.ipynb](AzureAISearchIndex.ipynb)


### Búsqueda semántica con Azure AI Search

Para probar la búsqueda semántica podemos buscar la frase "Resume la hoja de vida de Diego Zumárraga Mera" y ver cómo Azure AI Search puede devolver resultados relevantes.

```c#
const string MODEL_NAME = "openai/text-embedding-3-small";
var openAIOptions = new OpenAIClientOptions()
{
    Endpoint = new Uri("https://models.github.ai/inference")
};
EmbeddingClient embeddingClient = new(MODEL_NAME, new ApiKeyCredential(githubToken), openAIOptions);

string queryInput = "Resume la hoja de vida de Diego Zumárraga Mera";
OpenAIEmbeddingCollection embeddingResponse = embeddingClient.GenerateEmbeddings(new List<string> { queryInput });
var queryEmbedding = embeddingResponse.FirstOrDefault().ToFloats().ToArray();
Console.WriteLine($"Embedding length: {queryEmbedding.Length.ToString()}");

var searchClient = new SearchClient(new Uri(endpoint), INDEX_NAME, credential);

var vectorQuery = new VectorizedQuery(queryEmbedding)
{
    Fields = { "embedding" },
    KNearestNeighborsCount = 3
};

var options = new SearchOptions()
{
    QueryType = SearchQueryType.Semantic,
    SemanticSearch = new()
                {
                    SemanticConfigurationName = "default",
                    QueryCaption = new(QueryCaptionType.Extractive),
                    QueryAnswer = new(QueryAnswerType.Extractive)
                },
    Size = 2
};

options.VectorSearch = new();
options.VectorSearch.Queries.Add(vectorQuery);

Console.WriteLine("Buscando resultados...");
var response = searchClient.Search<Dictionary<string, object>>(queryInput, options);

string rawResults = response.GetRawResponse().Content.ToString();
Console.WriteLine($"Raw results: {rawResults}");
```

Ahora veamos el mejor resultado de la búsqueda:

```c#
var dicResults = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(rawResults);

object objAnswers; // List<Dictionary<string, object>>;
dicResults.TryGetValue("@search.answers", out objAnswers);

var answers = System.Text.Json.JsonSerializer.Deserialize<List<Dictionary<string, object>>>(objAnswers?.ToString() ?? string.Empty);
Console.WriteLine($"Number of answers: {answers?.Count ?? 0}");

if (answers != null && answers.Any())
{
    Console.WriteLine("Answers:");
    foreach (var answer in answers)
    {
        Console.WriteLine($"Text: {answer["text"]}");
        Console.WriteLine($"Score: {answer["score"]}");
    }
}
```
```
Answers:
Answer: Hoja de Vida: Diego Zumárraga Mera. Ingeniero de software con experiencia en desarrollo de aplicaciones web y móviles, apasionado por la inteligencia artificial y el aprendizaje automático.
Confidence: 0.9860000014305115
```
código completo en el notebook: [AzureAISearchQueryNet.ipynb](AzureAISearchQuery.ipynb)

Como podemos ver, Azure AI Search ha devuelto un resultado relevante para la consulta, a pesar de que en la consulta se incluye la frase común **"Hoja de Vida"**. Esto se debe a que hemos configurado la búsqueda semántica para que tenga en cuenta el título y las palabras clave del documento, lo que mejora la relevancia de los resultados.

Blog Post: [Azure AI Search: Cómo optimizar la búsqueda para tus aplicaciones RAG](https://acelera.tech/2025/06/15/azure-ai-search-como-optimizar-la-busqueda-para-tus-aplicaciones-rag/)
