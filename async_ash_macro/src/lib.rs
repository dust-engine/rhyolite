extern crate proc_macro;
use quote::ToTokens;
use syn::{
    parse::{Parse, ParseStream},
    spanned::Spanned,
    ExprUnsafe,
};

struct ExprGpuAsync {
    pub stmts: Vec<syn::Stmt>,
}
impl Parse for ExprGpuAsync {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(ExprGpuAsync {
            stmts: syn::Block::parse_within(input)?,
        })
    }
}


fn transform_stmt_asyncs(stmt: &syn::Stmt, async_transform: &mut impl FnMut(&syn::ExprAwait) -> syn::Expr) -> syn::Stmt {
    use syn::{Block, Expr, Pat, Stmt};
    fn transform_block(block: &Block, async_transform: &mut impl FnMut(&syn::ExprAwait) -> syn::Expr) -> Block {
        Block {
            brace_token: block.brace_token.clone(),
            stmts: block.stmts.iter().map(|stmt| transform_stmt(stmt, async_transform)).collect(),
        }
    }
    fn transform_pattern(pat: &Pat, async_transform: &mut impl FnMut(&syn::ExprAwait) -> syn::Expr) -> Pat {
        match pat {
            Pat::Box(pat) => Pat::Box(syn::PatBox {
                pat: Box::new(transform_pattern(&pat.pat, async_transform)),
                ..pat.clone()
            }),
            Pat::Ident(ident) => Pat::Ident(syn::PatIdent {
                subpat: ident.subpat.as_ref().map(|(at, subpat)| {
                    (at.clone(), Box::new(transform_pattern(subpat, async_transform)))
                }),
                ..ident.clone()
            }),
            Pat::Lit(lit) => Pat::Lit(syn::PatLit {
                expr: Box::new(transform_expr(&lit.expr, async_transform)),
                ..lit.clone()
            }),
            Pat::Or(clause) => Pat::Or(syn::PatOr {
                cases: clause.cases.iter().map(|pat| transform_pattern(pat, async_transform)).collect(),
                ..clause.clone()
            }),
            Pat::Range(range) => Pat::Range(syn::PatRange {
                lo: Box::new(transform_expr(&range.lo, async_transform)),
                hi: Box::new(transform_expr(&range.hi, async_transform)),
                ..range.clone()
            }),
            Pat::Reference(r) => Pat::Reference(syn::PatReference {
                pat: Box::new(transform_pattern(&r.pat, async_transform)),
                ..r.clone()
            }),
            Pat::Slice(slice) => Pat::Slice(syn::PatSlice {
                elems: slice.elems.iter().map(|pat| transform_pattern(pat, async_transform)).collect(),
                ..slice.clone()
            }),
            Pat::Struct(s) => Pat::Struct(syn::PatStruct {
                fields: s
                    .fields
                    .iter()
                    .map(|f| syn::FieldPat {
                        pat: Box::new(transform_pattern(&f.pat, async_transform)),
                        ..f.clone()
                    })
                    .collect(),
                ..s.clone()
            }),
            Pat::Tuple(tuple) => Pat::Tuple(syn::PatTuple {
                elems: tuple.elems.iter().map(|pat| transform_pattern(pat, async_transform)).collect(),
                ..tuple.clone()
            }),
            Pat::TupleStruct(tuple) => Pat::TupleStruct(syn::PatTupleStruct {
                pat: syn::PatTuple {
                    elems: tuple.pat.elems.iter().map(|pat| transform_pattern(pat, async_transform)).collect(),
                    ..tuple.pat.clone()
                },
                ..tuple.clone()
            }),
            Pat::Type(ty) => Pat::Type(syn::PatType {
                pat: Box::new(transform_pattern(&ty.pat, async_transform)),
                ..ty.clone()
            }),
            _ => pat.clone(),
        }
    }
    fn transform_expr(expr: &Expr, async_transform: &mut impl FnMut(&syn::ExprAwait) -> syn::Expr) -> Expr {
        match expr {
            Expr::Array(arr) => Expr::Array(syn::ExprArray {
                elems: arr.elems.iter().map(|expr| transform_expr(expr, async_transform)).collect(),
                ..arr.clone()
            }),
            Expr::Assign(assign) => Expr::Assign(syn::ExprAssign {
                left: Box::new(transform_expr(&*assign.left, async_transform)),
                right: Box::new(transform_expr(&*assign.right, async_transform)),
                ..assign.clone()
            }),
            Expr::AssignOp(assign) => Expr::AssignOp(syn::ExprAssignOp {
                left: Box::new(transform_expr(&*assign.left, async_transform)),
                right: Box::new(transform_expr(&*assign.right, async_transform)),
                ..assign.clone()
            }),
            Expr::Binary(binary) => Expr::Binary(syn::ExprBinary {
                left: Box::new(transform_expr(&*binary.left, async_transform)),
                right: Box::new(transform_expr(&*binary.right, async_transform)),
                ..binary.clone()
            }),
            Expr::Block(block) => Expr::Block(syn::ExprBlock {
                block: transform_block(&block.block, async_transform),
                ..block.clone()
            }),
            Expr::Call(call) => Expr::Call(syn::ExprCall {
                func: Box::new(transform_expr(&*call.func, async_transform)),
                args: call.args.iter().map(|expr| transform_expr(expr, async_transform)).collect(),
                ..call.clone()
            }),
            Expr::Cast(cast) => Expr::Cast(syn::ExprCast {
                expr: Box::new(transform_expr(&*cast.expr, async_transform)),
                ..cast.clone()
            }),
            Expr::Field(field) => Expr::Field(syn::ExprField {
                base: Box::new(transform_expr(&*field.base, async_transform)),
                ..field.clone()
            }),
            Expr::ForLoop(l) => Expr::ForLoop(syn::ExprForLoop {
                pat: transform_pattern(&l.pat, async_transform),
                expr: Box::new(transform_expr(&*l.expr, async_transform)),
                body: transform_block(&l.body, async_transform),
                ..l.clone()
            }),
            Expr::Group(group) => Expr::Group(syn::ExprGroup {
                expr: Box::new(transform_expr(&group.expr, async_transform)),
                ..group.clone()
            }),
            Expr::If(if_stmt) => Expr::If(syn::ExprIf {
                cond: Box::new(transform_expr(&if_stmt.cond, async_transform)),
                then_branch: transform_block(&if_stmt.then_branch, async_transform),
                else_branch: if_stmt.else_branch.as_ref().map(|(else_token, expr)| {
                    (else_token.clone(), Box::new(transform_expr(&expr, async_transform)))
                }),
                ..if_stmt.clone()
            }),
            Expr::Index(index_expr) => Expr::Index(syn::ExprIndex {
                expr: Box::new(transform_expr(&*index_expr.expr, async_transform)),
                index: Box::new(transform_expr(&*index_expr.index, async_transform)),
                ..index_expr.clone()
            }),
            Expr::Let(l) => Expr::Let(syn::ExprLet {
                pat: transform_pattern(&l.pat, async_transform),
                expr: Box::new(transform_expr(&*l.expr, async_transform)),
                ..l.clone()
            }),
            Expr::Loop(loop_stmt) => Expr::Loop(syn::ExprLoop {
                body: transform_block(&loop_stmt.body, async_transform),
                ..loop_stmt.clone()
            }),
            Expr::Match(match_expr) => Expr::Match(syn::ExprMatch {
                expr: Box::new(transform_expr(&*match_expr.expr, async_transform)),
                arms: match_expr
                    .arms
                    .iter()
                    .map(|arm| syn::Arm {
                        pat: transform_pattern(&arm.pat, async_transform),
                        guard: arm.guard.as_ref().map(|(guard_token, expr)| {
                            (guard_token.clone(), Box::new(transform_expr(&expr, async_transform)))
                        }),
                        body: Box::new(transform_expr(&arm.body, async_transform)),
                        ..arm.clone()
                    })
                    .collect(),
                ..match_expr.clone()
            }),
            Expr::MethodCall(call) => Expr::MethodCall(syn::ExprMethodCall {
                receiver: Box::new(transform_expr(&call.receiver, async_transform)),
                args: call.args.iter().map(|expr| transform_expr(expr, async_transform)).collect(),
                ..call.clone()
            }),
            Expr::Paren(paren) => Expr::Paren(syn::ExprParen {
                expr: Box::new(transform_expr(&paren.expr, async_transform)),
                ..paren.clone()
            }),
            Expr::Range(range) => Expr::Range(syn::ExprRange {
                from: range.from.as_ref().map(|f| Box::new(transform_expr(&f, async_transform))),
                to: range.to.as_ref().map(|t| Box::new(transform_expr(&t, async_transform))),
                ..range.clone()
            }),
            Expr::Reference(reference) => Expr::Reference(syn::ExprReference {
                expr: Box::new(transform_expr(&reference.expr, async_transform)),
                ..reference.clone()
            }),
            Expr::Repeat(repeat) => Expr::Repeat(syn::ExprRepeat {
                expr: Box::new(transform_expr(&repeat.expr, async_transform)),
                len: Box::new(transform_expr(&repeat.len, async_transform)),
                ..repeat.clone()
            }),
            Expr::Return(ret) => Expr::Return(syn::ExprReturn {
                expr: ret.expr.as_ref().map(|e| Box::new(transform_expr(&e, async_transform))),
                ..ret.clone()
            }),
            Expr::Struct(s) => Expr::Struct(syn::ExprStruct {
                fields: s
                    .fields
                    .iter()
                    .map(|f| syn::FieldValue {
                        member: f.member.clone(),
                        expr: transform_expr(&f.expr, async_transform),
                        ..f.clone()
                    })
                    .collect(),
                ..s.clone()
            }),
            Expr::Try(s) => Expr::Try(syn::ExprTry {
                expr: Box::new(transform_expr(&s.expr, async_transform)),
                ..s.clone()
            }),
            Expr::Tuple(tuple) => Expr::Tuple(syn::ExprTuple {
                elems: tuple.elems.iter().map(|expr| transform_expr(expr, async_transform)).collect(),
                ..tuple.clone()
            }),
            Expr::Type(type_expr) => Expr::Type(syn::ExprType {
                expr: Box::new(transform_expr(&type_expr.expr, async_transform)),
                ..type_expr.clone()
            }),
            Expr::Unary(unary) => Expr::Unary(syn::ExprUnary {
                expr: Box::new(transform_expr(&unary.expr, async_transform)),
                ..unary.clone()
            }),
            Expr::Unsafe(unsafe_stmt) => Expr::Unsafe(ExprUnsafe {
                block: transform_block(&unsafe_stmt.block, async_transform),
                ..unsafe_stmt.clone()
            }),
            Expr::While(while_stmt) => Expr::While(syn::ExprWhile {
                cond: Box::new(transform_expr(&while_stmt.cond, async_transform)),
                body: transform_block(&while_stmt.body, async_transform),
                ..while_stmt.clone()
            }),
            Expr::Await(await_expr) => (async_transform)(await_expr),
            _ => expr.clone(),
        }
    }
    fn transform_stmt(stmt: &Stmt, async_transform: &mut impl FnMut(&syn::ExprAwait) -> Expr) -> Stmt {
        match stmt {
            Stmt::Local(local) => Stmt::Local(syn::Local {
                pat: transform_pattern(&local.pat, async_transform),
                init: local.init.as_ref().map(|(eq_token, expr)| {
                    (eq_token.clone(), Box::new(transform_expr(&expr, async_transform)))
                }),
                ..local.clone()
            }),
            Stmt::Expr(expr) => Stmt::Expr(transform_expr(expr, async_transform)),
            Stmt::Semi(expr, semi) => Stmt::Semi(transform_expr(expr, async_transform), semi.clone()),
            _ => stmt.clone(),
        }
    }
    transform_stmt(stmt, async_transform)
}

#[proc_macro]
pub fn gpu(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro_gpu(input.into()).into()
}


fn proc_macro_gpu(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<ExprGpuAsync>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };

    let mut future_count: u32 = 0;
    let mut collect_futures_block_stmts = Vec::with_capacity(input.stmts.len() + 1);
    collect_futures_block_stmts.extend(input
        .stmts
        .iter()
        .map(|stmt| transform_stmt_asyncs(stmt, &mut |expr: &syn::ExprAwait| {
            let future_ident = syn::Ident::new(&format!("__future_{}", future_count), proc_macro2::Span::call_site());
            future_count += 1;
            syn::Expr::Assign(syn::ExprAssign{
                attrs: Vec::new(),
                left: Box::new(syn::Expr::Path(syn::ExprPath{
                    attrs: Vec::new(),
                    qself: None,
                    path: syn::Path{
                        leading_colon: None,
                        segments: std::iter::once(syn::PathSegment{
                            ident: future_ident.clone(),
                            arguments: syn::PathArguments::None
                        }).collect()
                    }
                })),
                eq_token: Default::default(),
                right: expr.base.clone()
            })
        })));
    collect_futures_block_stmts.push(syn::Stmt::Expr(syn::Expr::Tuple(syn::ExprTuple {
        elems: (0..future_count).map(|i| {
            syn::Expr::Path(syn::ExprPath {
            attrs: vec![],
            qself: None,
            path: syn::Path {
                leading_colon: None,
                segments: std::iter::once(syn::PathSegment {
                    ident: syn::Ident::new(&format!("__future_{}", i), proc_macro2::Span::call_site()),
                    arguments: syn::PathArguments::None,
                }).collect()
            },
        })
        }).collect(),
        attrs: Vec::new(),
        paren_token: Default::default(),
    })));


    let collect_futures_block = syn::Block {
        brace_token: Default::default(),
        stmts: collect_futures_block_stmts,
    };

    quote::quote! {
        #collect_futures_block
    }
    .into()
}

mod tests {
    #[test]
    fn test_basic() {
        let quote = quote::quote! {
            let aaa = copy_buffer().await;
        };
        let parsed = super::proc_macro_gpu(quote);
        println!("{:?}", parsed.to_string());
    }
}
