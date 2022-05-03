# mood

This is the repository for mood detection. The input is text, and the output is the interpreted mood of the text. This is done by using an LSTM based model. The model is then exposed over a web server.

## API usage example

```bash
curl -H "Content-Type: application/json" -X POST -d '{"userInput":"i am happy"}' https://scrawlmood.herokuapp.com/api
```

## license 

[MIT](./LICENSE)

