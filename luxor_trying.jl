using Luxor
Drawing()
origin()
setcolor("red")
circle(Point(0,0), 100, :fill)
finish()
preview()
ngon(Point(0, 0), 80, 3, vertices=true)


function test()
begin
Drawing(1000,1000)
cells = Table(10,10)
setcolor("red")
for i=1:10, j=1:10
    println(cells[i,j])
    box(cells[i,j], 100,100, :fill)
    if i>3
        setcolor("blue")
    end
end
end
finish()
preview()
end

begin
    Drawing(1000,1000)
    background("cyan")
    setcolor("red")
    origin()
    cells = Table(10,10, 100, 100)
    for i=1:10, j=1:10
        println(cells[i,j])
        box(cells[i,j], 100,100, :fill)
        if i>3
            setcolor("blue")
        end
    end
    finish()
    preview()
end