//获取图片信息
function detect(img_base64) {
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/ai/detect', false)
    var data = new FormData()
    data.append("img", img_base64)
    xhr.send(data)
    return xhr.response
}
function test(testText) {
    var xhr = new XMLHttpRequest()
    xhr.open('POST', '/ai/test', false)
    var data = new FormData()
    data.append("img", testText)
    xhr.send(data)
    console.log(xhr.response)
}
function selectImage(input){
    var file = input.files[0]
    const image = new Image()
    image.src = URL.createObjectURL(file)
    document.getElementById("img_window").src=image.src
    image.onload = function (event){
        URL.revokeObjectURL(this.src)
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d")
        canvas.height = image.height
        canvas.width = image.width
        ctx.drawImage(image, 0, 0)
        const response = detect(canvas.toDataURL("image/jpeg", 1.0));
        console.log(response)
        parseResponse(response)
    }

}
function parseResponse(response){
    var table = document.getElementById("result_table")
    var res = JSON.parse(response)
    table.deleteRow(0)
    best_line = table.insertRow(0)
    best_line.insertCell(0).innerHTML="最佳结果:"+res.bestMatch
    best_line.insertCell(1).innerHTML="l2范数:"+res.norm
    var arr = res.candidates
    for(i=0;i<arr.length;i++){
        line = table.insertRow(i+1)
        line.insertCell(0).innerHTML="备选结果:"+arr[i].character
        line.insertCell(1).innerHTML="l2范数:"+arr[i].norm
    }
    document.getElementById("file_selector").value=null
}